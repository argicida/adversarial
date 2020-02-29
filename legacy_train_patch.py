"""
Training code for Adversarial patch training


"""
from cli_config import FLAGS
import legacy_patch_config
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision import transforms

from tensorboardX import SummaryWriter

from torch.utils.data import Dataset
from inria import LegacyYolov2InriaDataset
from patch_utilities import Patch, PatchApplier, PatchTransformer, TotalVariation, NPSCalculator, SaturationCalculator

#yolov2
from darknet import Darknet

#yolov3
from implementations.yolov3.models import Darknet as Yolov3

#ssd
from implementations.ssd.vision.ssd.vgg_ssd import create_vgg_ssd

# import datetime


class Yolov2_Output_Extractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(Yolov2_Output_Extractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf


class Patch(nn.Module):
    def __init__(self, patch_size, typ="grey", tanh=True):
        super(Patch, self).__init__()
        if typ == 'grey':
            # when params are 0. the rgbs are 0.5
            self.params = nn.Parameter.__new__(torch.full((3, patch_size, patch_size), 0))
        elif typ == 'random':
            # uniform distribution range from -2 to -2
            self.params = nn.Parameter.__new__((torch.rand((3, patch_size, patch_size))*2 - 1) * 2)
        # both options force the patch to have valid rgb values
        if tanh:
            self.constraint = self.tanh_constraint
        else:
            self.constraint = self.sigmoid_constraint

    def tanh_constraint(self, params):
        return 0.5 * (torch.tanh(params) + 1)

    def sigmoid_constraint(self, params):
        return torch.sigmoid(params)

    def forward(self):
        return self.constraint(self.params)


def load_yolov2(device=0):
    yolov2_cfgfile = "cfg/yolov2.cfg"
    yolov2_weightfile = "weights/yolov2.weights"
    yolov2 = Darknet(yolov2_cfgfile)
    yolov2.load_weights(yolov2_weightfile)
    return yolov2.eval().cuda(device)


def load_yolov3(device=0):
    yolov3_cfgfile = "./implementations/yolov3/config/yolov3.cfg"
    yolov3_weightfile = "./implementations/yolov3/weights/yolov3.weights"
    yolov3 = Yolov3(yolov3_cfgfile)
    yolov3.load_darknet_weights(yolov3_weightfile)
    return yolov3.eval().cuda(device)


def load_ssd(device=0):
    ssd_weightfile = "./implementations/ssd/models/vgg16-ssd-mp-0_7726.pth"
    voc_num_classes = 21
    # returns a nn.module
    # setting is_test to false since we dont need boxes, just confidence scores
    ssd = create_vgg_ssd(voc_num_classes, is_test=False)
    ssd.load(ssd_weightfile)
    return ssd.eval().cuda(device)


class Yolov3_Output_Extractor(nn.Module):
    def __init__(self, cls_id, num_cls, config):
        super(Yolov3_Output_Extractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.confidence_thresh = 0.5
        self.nms_thresh = 0.4
        return

    def forward(self, detections):
        object_conf = detections[:,:,4]
        class_conf = detections[:,:,self.cls_id + 5]
        picked = self.config.loss_target(object_conf, class_conf)
        max_conf, _ = torch.max(picked, dim=1)
        return max_conf


class SSD_Output_Extractor(nn.Module):
    def __init__(self, cls_id):
        super(SSD_Output_Extractor, self).__init__()
        self.cls_id = cls_id
        return

    def forward(self, ssd_out):
        # dim confidence: batch, num_priors, num_classes
        # dim locations: batch, num_priors, 4
        confidence, locations = ssd_out

        ### following approach minimize all confidences at human detections
        confidence_scores = F.softmax(confidence, dim=2)
        relevant_human_mask = (confidence_scores[:, :, self.cls_id] > 0.1).float()
        relevant_confidence = confidence * relevant_human_mask.unsqueeze(-1)
        return torch.sum(relevant_confidence)

        ### following approaches only extract human confidence logits
        #relevant_confidence = confidence[:, :, self.cls_id]
        #mean_confidence = torch.mean(relevant_confidence, dim=1)
        #return mean_confidence
        #max_confidence, _ = torch.max(relevant_confidence, dim=1)
        #return max_confidence
        #return torch.sum(relevant_confidence, dim=1)

        ### following approaches extract the margin between human and other classes
        ### then run a targeted attack that minimizes the margin between human and nearest class logits
        # num_classes = confidence.shape[-1]
        # possible_targets_mask = np.ones(num_classes, dtype=bool)
        # possible_targets_mask[self.cls_id] = False
        # dim total confidence: batch, num_classes
        #class_total_confidences = torch.sum(confidence, dim=1)
        # dim nearest_targets: batch, 1
        #nearest_targets, _ = torch.max(class_total_confidences[:,possible_targets_mask], dim=1)
        # dim human_total_confidences: batch, 1
        #human_total_confidences = class_total_confidences[:, self.cls_id]
        #return human_total_confidences - nearest_targets
        # human_confidences = confidence[:, :, self.cls_id]
        # possible_target_confidences = confidence[:, :, possible_targets_mask]
        # target_confidences, _ = torch.max(possible_target_confidences, dim=2)
        # dim margins: batch, num_priors
        # stops optimizing as soon as another class has bigger logits by magnitude of m
        #m = 5
        #margins = F.relu(human_confidences - target_confidences + m)
        # margins = human_confidences - target_confidences
        # confidence_scores = F.softmax(confidence, dim=2)
        # relevant_human_mask = (confidence_scores[:, :, self.cls_id] > 0.1).float()
        # margins = margins * relevant_human_mask
        # return torch.sum(margins, dim=1)


        ### following approach uses cross entropy on human confidences
        # dim: batch, num_priors
        #human_logits = confidence[:, :, self.cls_id]
        #zeros = torch.zeros(human_logits.shape).cuda()
        #loss_fun = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()
        # dim: batch, num_priors
        #loss = loss_fun(human_logits, zeros)
        #return torch.sum(loss, dim=1)


# Checks for the correct input length and then runs the trainer
def main():
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    config = legacy_patch_config.patch_configs["paper_obj"]()
    data_img_dir = "inria/Train/pos"
    data_lab_dir = "inria/Train/pos/yolo-labels"
    printable_vals_file = "non_printability/30values.txt"
    patch_size = 300

    # yolov2 = load_yolov2()
    yolov3 = load_yolov3()
    ssd = load_ssd()

    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    # yolov2_output_extractor = Yolov2_Output_Extractor(0, 80, config).cuda()
    yolov3_output_extractor = Yolov3_Output_Extractor(0, 80, config).cuda()
    ssd_output_extractor = SSD_Output_Extractor(15)
    non_printability_calculator = NPSCalculator(printable_vals_file, patch_size).cuda()
    total_variation = TotalVariation().cuda()
    saturation_calculator = SaturationCalculator().cuda()
    # Property in which most data is written to, including the patch
    writer = SummaryWriter(logdir=FLAGS.log_dir)

    # Initialize some settings
    img_size = 608  # dataloader configured with dimensions from yolov2
    batch_size = FLAGS.bs
    max_box_per_image = FLAGS.max_labs

    # Generate stating point
    patch_module = Patch(patch_size=config.patch_size, typ=FLAGS.start_patch, tanh=True).cuda()
    # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

    # Sets up training and determines how long the training length will be
    train_loader = torch.utils.data.DataLoader(LegacyYolov2InriaDataset(config.img_dir, config.lab_dir,
                                                                        max_box_per_image, img_size,
                                                                        shuffle=True),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=10)

    # Sets the epoch length
    epoch_length = len(train_loader)
    print(f'One epoch is {epoch_length}')

    # Creates the object which will optimize the patch, and sets the learning rate
    optimizer = optim.Adam(patch_module.parameters(), lr=FLAGS.lr)

    # Schedules tasks on the gpu to optimize performance
    scheduler = config.scheduler_factory(optimizer)

    et0 = time.time()
    for epoch in range(FLAGS.n_epochs):
        # for epoch in range(1):
        # Sets the gradient inputs to zero
        ep_det_loss = 0
        ep_nps_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        # ep_ssd_loss = 0
        # ep_yolov2_loss = 0
        ep_yolov3_loss = 0
        bt0 = time.time()

        # I have no fucking clue how long this is supposed to be running, probably for the epoch length? Needs
        # More research
        # TODO: note from Perry: yes this enumerates through some sort of file iterator for number of epochs
        #         the tqdm shit is just for progress bar, except for the total argument
        for i_batch, (img_batch, gt_boxes_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                         total=epoch_length):
            with autograd.detect_anomaly():
                # Optimizes everything to run on GPUs
                img_batch = img_batch.cuda()
                gt_boxes_batch_cpu = gt_boxes_batch
                gt_boxes_batch = gt_boxes_batch.cuda()
                # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                adv_patch = patch_module()

                # Creates a patch transformer with a default grey patch. Can't find this object anywhere but most
                # Likely it allows the patch to be moved around on inputted images.
                # TODO: Find documentation for this object
                adv_batch_t = patch_transformer(adv_patch, gt_boxes_batch, img_size, do_rotate=True, rand_loc=False)

                # Can't find this object anywhere else, most likely allows the patch to be put into inputted photos
                # TODO: find documentation for this object
                # TODO: note from Perry: which object? patch_applier is from this file, F is torch.nn.functional
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                # p_img_batch = F.interpolate(p_img_batch, (yolov2.height, yolov2.width))
                ssd_p_img_batch = F.interpolate(p_img_batch, (300, 300))
                ssd_p_img_batch = ssd_p_img_batch * 255  # "de"-normalize
                ssd_p_img_batch[:, 0, :, :] -= 123
                ssd_p_img_batch[:, 1, :, :] -= 117
                ssd_p_img_batch[:, 2, :, :] -= 104

                # TODO: Figure out exactly what these transforms are doing
                # TODO: note from Perry: these two lines seem to only be used for debugging purposes, commenting them out
                # img = p_img_batch[1, :, :,]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()

                # Looks to be where the given patch is evaluated, however all these objects are not included within
                # The given darknet model. Documentation for the original needs to be found and researched
                # output_yolov2 = yolov2(p_img_batch)
                output_yolov3 = yolov3(p_img_batch)
                # output_ssd = ssd(ssd_p_img_batch)
                # max_prob_yolov2 = yolov2_output_extractor(output_yolov2)
                max_prob_yolov3 = yolov3_output_extractor(output_yolov3)
                # max_prob_ssd = ssd_output_extractor(output_ssd)

                non_printability_score = non_printability_calculator(adv_patch)
                patch_variation = total_variation(adv_patch)
                # patch_saturation = saturation_calculator(adv_patch)

                # Calculates the loss in the new patch, then mashes them all together
                printability_loss = non_printability_score * 0.01
                patch_variation_loss = patch_variation * 2.5
                # patch_saturation_loss = patch_saturation*1

                # detection_loss_yolov2 = torch.mean(max_prob_yolov2)
                detection_loss_yolov3 = torch.mean(max_prob_yolov3)
                # detection_loss_ssd = torch.mean(max_prob_ssd)

                # detection_loss = detection_loss_yolov2 + detection_loss_yolov3 + detection_loss_ssd
                detection_loss = detection_loss_yolov3 * 1
                loss = torch.tensor(0)
                loss = detection_loss
                # loss = loss + printability_loss
                # loss = loss + torch.max(patch_variation_loss, torch.tensor(0.1).cuda())
                # loss = loss + patch_saturation_loss

                # for debugging purposes
                ep_det_loss += detection_loss.detach().cpu().numpy()
                ep_nps_loss += printability_loss.detach().cpu().numpy()
                ep_tv_loss += patch_variation_loss.detach().cpu().numpy()
                # ep_ssd_loss += detection_loss_ssd.detach().cpu().numpy()
                # ep_yolov2_loss += detection_loss_yolov2.detach().cpu().numpy()
                ep_yolov3_loss += detection_loss_yolov3.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                # Calculates the gradient of the loss function
                loss.backward()

                # plot_grad_flow([("adv_patch_cpu",adv_patch_cpu)])
                # plt.show()

                # for debugging backprop of target losses
                mean_absolute_gradient = torch.mean(torch.abs(patch_module.params.grad))
                max_absolute_gradient = torch.max(torch.abs(patch_module.params.grad))

                # Performs one step in optimization of the patch
                optimizer.step()

                # Clears all gradients after each step. Default is to accumulate them, we don't want that
                optimizer.zero_grad()
                # adv_patch_cpu.data.clamp_(0,1)       # keep patch in image range # not needed due to patch module

                bt1 = time.time()
                iteration = epoch_length * epoch + i_batch

                # Writes all this data to the object's tensorboard item, which was initialized as 'writer'
                writer.add_scalar('batch/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('batch/total_det_loss', detection_loss.detach().cpu().numpy(), iteration)
                # writer.add_scalar('batch/YOLOv2_loss', detection_loss_yolov2.detach().cpu().numpy(), iteration)
                # writer.add_scalar('batch/YOLOv3_loss', detection_loss_yolov3.detach().cpu().numpy(), iteration)
                # writer.add_scalar('batch/SSD_loss', detection_loss_ssd.detach().cpu().numpy(), iteration)
                writer.add_scalar('batch/printability_loss', printability_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('batch/tv_loss', patch_variation_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('batch/epoch', epoch, iteration)
                writer.add_scalar('batch/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar("batch/mean_absolute_gradient", mean_absolute_gradient.detach().cpu().numpy(),
                                  iteration)
                writer.add_scalar("batch/max_absolute_gradient", max_absolute_gradient.detach().cpu().numpy(),
                                  iteration)

                # If the training is over, add an endline character, else clearn the following variables
                if i_batch + 1 >= len(train_loader):
                    print('\n')
                else:
                    # del adv_batch_t, output_yolov2, max_prob_yolov2, detection_loss_yolov2, p_img_batch, printability_loss, patch_variation_loss, loss
                    torch.cuda.empty_cache()
                bt0 = time.time()

        # Calculate average loss over the course of training
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(train_loader)
        ep_nps_loss = ep_nps_loss / len(train_loader)
        ep_tv_loss = ep_tv_loss / len(train_loader)
        ep_loss = ep_loss / len(train_loader)
        # ep_yolov2_loss = ep_yolov2_loss/len(train_loader)
        ep_yolov3_loss = ep_yolov3_loss / len(train_loader)
        # ep_ssd_loss = ep_ssd_loss/len(train_loader)
        writer.add_scalar('loss/total_loss', ep_loss, epoch)
        writer.add_scalar('loss/total_det_loss', ep_det_loss, epoch)
        # writer.add_scalar('loss/YOLOv2_loss', ep_yolov2_loss, epoch)
        # writer.add_scalar('loss/YOLOv3_loss', ep_yolov3_loss, epoch)
        # writer.add_scalar('loss/SSD_loss', ep_ssd_loss, epoch)
        writer.add_scalar('loss/printability_loss', ep_nps_loss, epoch)
        writer.add_scalar('loss/tv_loss', ep_tv_loss, epoch)

        # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        # plt.imshow(im)
        # plt.savefig(f'pics/{time_str}_{config.patch_name}_{epoch}.png')
        # Output statistics and training time
        scheduler.step(ep_loss)

        print('  EPOCH NR: ', epoch),
        print('EPOCH LOSS: ', ep_loss)
        print('  DET LOSS: ', ep_det_loss)
        print('  NPS LOSS: ', ep_nps_loss)
        print('   TV LOSS: ', ep_tv_loss)
        print('EPOCH TIME: ', et1 - et0)
        # del adv_batch_t, output_yolov2, max_prob_yolov2, detection_loss_yolov2, p_img_batch, printability_loss, patch_variation_loss, loss
        # torch.cuda.empty_cache()
        writer.add_image('patch', adv_patch.detach().cpu().numpy(), epoch)

        et0 = time.time()

    # At the end of training, save image
    im = transforms.ToPILImage('RGB')(adv_patch.cpu())
    # Specifies file to save trained patch to
    im.save(os.path.join(FLAGS.log_dir, "patch.png"), "PNG")


if __name__ == '__main__':
    main()


