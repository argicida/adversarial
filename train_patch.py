"""
Training code for Adversarial patch training


"""

# import PIL
# import load_data
from tqdm import tqdm

from implementations.yolov3.models import Darknet as Yolov3
from implementations.yolov3.utils import utils as yolov3_utils

from implementations.ssd.vision.ssd.vgg_ssd import create_vgg_ssd

from load_data import *
# import gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
# import subprocess

import patch_config
import sys
import time
# import datetime


class Patch(nn.Module):
    def __init__(self, patch_size, typ="grey", tanh=True):
        super(Patch, self).__init__()
        if typ == 'grey':
            # when params are 0. the rgbs are 0.5
            self.params = nn.Parameter(torch.full((3, patch_size, patch_size), 0))
        elif typ == 'random':
            # uniform distribution range from -2 to -2
            self.params = nn.Parameter((torch.rand((3, patch_size, patch_size))*2 - 1) * 2)
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


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


class Yolov3_Output_Extractor(nn.Module):
    def __init__(self, cls_id, num_cls, config):
        super(Yolov3_Output_Extractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.confidence_thresh = 0.5
        self.nms_thresh = 0.4
        return

    def forward(self, v3_out):
        with torch.no_grad():
            outputs = yolov3_utils.non_max_suppression(v3_out)
       	return outputs


class SSD_Output_Extractor(nn.Module):
    def __init__(self, cls_id):
        super(SSD_Output_Extractor, self).__init__()
        self.cls_id = cls_id
        return

    def forward(self, ssd_out):
        # dim confidence: batch, num_priors, num_classes
        # dim locations: batch, num_priors, 4
        confidence, locations = ssd_out

        ### following approaches only extract human confidence logits
        #relevant_confidence = confidence[:, :, self.cls_id]
        #mean_confidence = torch.mean(relevant_confidence, dim=1)
        #return mean_confidence
        #max_confidence, _ = torch.max(relevant_confidence, dim=1)
        #return max_confidence
        #return torch.sum(relevant_confidence, dim=1)

        ### following approach extract the margin between human and other classes
        ### then run a targeted attack that minimizes the margin between human and nearest class logits
        # dim total confidence: batch, num_classes
        class_total_confidences = torch.sum(confidence, dim=1)
        num_classes = class_total_confidences.shape[-1]
        non_target_mask = np.ones(num_classes, dtype=bool)
        non_target_mask[self.cls_id] = False
        # dim nearest_targets: batch, 1
        nearest_targets, _ = torch.max(class_total_confidences[:,non_target_mask], dim=1)
        # dim human_total_confidences: batch, 1
        human_total_confidences = class_total_confidences[:, self.cls_id]
        return human_total_confidences - nearest_targets


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        #self.yolov2 = load_yolov2()
        #self.yolov3 = load_yolov3()
        self.ssd = load_ssd()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        #self.yolov2_output_extractor = Yolov2_Output_Extractor(0, 80, self.config).cuda()
        #self.yolov3_output_extractor = Yolov3_Output_Extractor(0, 80, self.config).cuda()
        self.ssd_output_extractor = SSD_Output_Extractor(15)
        self.non_printability_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.saturation_calculator = SaturationCalculator().cuda()
        # Property in which most data is written to, including the patch
        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        #subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        """

        # Initialize some settings
        img_size = 608 # dataloader configured with dimensions from yolov2
        batch_size = self.config.batch_size
        n_epochs = 500
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        patch_module = Patch(patch_size=self.config.patch_size, typ=self.config.start_patch, tanh=True).cuda()
        # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")


        # Sets up training and determines how long the training length will be
        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

        # Sets the epoch length
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        # Creates the object which will optimize the patch, and sets the learning rate
        optimizer = optim.Adam(patch_module.parameters(), lr=self.config.start_learning_rate, weight_decay=self.config.decay, amsgrad=True)

        # Schedules tasks on the gpu to optimize performance
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
        #for epoch in range(1):
            # Sets the gradient inputs to zero
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_ssd_loss = 0
            #ep_yolov2_loss = 0
            #ep_yolov3_loss = 0
            bt0 = time.time()

            # I have no fucking clue how long this is supposed to be running, probably for the epoch length? Needs
            # More research
            # TODO: note from Perry: yes this enumerates through some sort of file iterator for number of epochs
            #         the tqdm shit is just for progress bar, except for the total argument
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    # Optimizes everything to run on GPUs
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = patch_module()

                    # Creates a patch transformer with a default grey patch. Can't find this object anywhere but most
                    # Likely it allows the patch to be moved around on inputted images.
                    # TODO: Find documentation for this object
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                    # Can't find this object anywhere else, most likely allows the patch to be put into inputted photos
                    # TODO: find documentation for this object
                    # TODO: note from Perry: which object? patch_applier is from this file, F is torch.nn.functional
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    #p_img_batch = F.interpolate(p_img_batch, (self.yolov2.height, self.yolov2.width))

                    ssd_p_img_batch = F.interpolate(p_img_batch, (300, 300))
                    ssd_p_img_batch[:, 0, :, :] -= 123
                    ssd_p_img_batch[:, 1, :, :] -= 117
                    ssd_p_img_batch[:, 2, :, :] -= 104

                    # TODO: Figure out exactly what these transforms are doing
                    # TODO: note from Perry: these two lines seem to only be used for debugging purposes, commenting them out
                    #img = p_img_batch[1, :, :,]
                    #img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    # Looks to be where the given patch is evaluated, however all these objects are not included within
                    # The given darknet model. Documentation for the original needs to be found and researched
                    #output_yolov2 = self.yolov2(p_img_batch)
                    #output_yolov3 = self.yolov3(p_img_batch)
                    output_ssd = self.ssd(ssd_p_img_batch)
                    #max_prob_yolov2 = self.yolov2_output_extractor(output_yolov2)
                    #max_prob_yolov3 = self.yolov3_output_extractor(output_yolov3)
                    max_prob_ssd= self.ssd_output_extractor(output_ssd)
                    #print("max_prob_yolov3[0].size(): " + str(max_prob_yolov3[0].size()))

                    non_printability_score = self.non_printability_calculator(adv_patch)
                    patch_variation = self.total_variation(adv_patch)
                    #patch_saturation = self.saturation_calculator(adv_patch)

                    # Calculates the loss in the new patch, then mashes them all together
                    printability_loss = non_printability_score*0.01
                    patch_variation_loss = patch_variation*2.5
                    #patch_saturation_loss = patch_saturation*1

                    #detection_loss_yolov2 = torch.mean(max_prob_yolov2)
                    #detection_loss_yolov3 = torch.mean(max_prob_yolov3)
                    detection_loss_ssd = torch.mean(max_prob_ssd)

                    #detectino_loss = detection_loss_yolov2 + detection_loss_yolov3 + detection_loss_ssd
                    detection_loss = detection_loss_ssd * 1
                    loss = 0
                    loss += detection_loss
                    #loss += printability_loss
                    #loss += torch.max(patch_variation_loss, torch.tensor(0.1).cuda())
                    #loss += patch_saturation_loss

                    # for debugging purposes
                    ep_det_loss += detection_loss.detach().cpu().numpy()
                    ep_nps_loss += printability_loss.detach().cpu().numpy()
                    ep_tv_loss += patch_variation_loss.detach().cpu().numpy()
                    ep_ssd_loss += detection_loss_ssd.detach().cpu().numpy()
                    #ep_yolov2_loss += detection_loss_yolov2.detach().cpu().numpy()
                    #ep_yolov3_loss += detection_loss_yolov3.detach().cpu().numpy()
                    ep_loss += loss.detach().cpu().numpy()

                    # Calculates the gradient of the loss function
                    loss.backward()

                    #plot_grad_flow([("adv_patch_cpu",adv_patch_cpu)])
                    #plt.show()

                    # for debugging backprop of target losses
                    mean_absolute_gradient = torch.mean(torch.abs(patch_module.params.grad))
                    max_absolute_gradient = torch.max(torch.abs(patch_module.params.grad))

                    # Performs one step in optimization of the patch
                    optimizer.step()

                    # Clears all gradients after each step. Default is to accumulate them, we don't want that
                    optimizer.zero_grad()
                    #adv_patch_cpu.data.clamp_(0,1)       # keep patch in image range # not needed due to patch module

                    bt1 = time.time()
                    iteration = self.epoch_length * epoch + i_batch

                    # Writes all this data to the object's tensorboard item, which was initialized as 'writer'
                    self.writer.add_scalar('batch/total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('batch/total_det_loss', detection_loss.detach().cpu().numpy(), iteration)
                    #self.writer.add_scalar('batch/YOLOv2_loss', detection_loss_yolov2.detach().cpu().numpy(), iteration)
                    #self.writer.add_scalar('batch/YOLOv3_loss', detection_loss_yolov3.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('batch/SSD_loss', detection_loss_ssd.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('batch/printability_loss', printability_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('batch/tv_loss', patch_variation_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('batch/epoch', epoch, iteration)
                    self.writer.add_scalar('batch/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                    self.writer.add_scalar("batch/mean_absolute_gradient", mean_absolute_gradient.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar("batch/max_absolute_gradient", max_absolute_gradient.detach().cpu().numpy(), iteration)

                    # If the training is over, add an endline character, else clearn the following variables
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        #del adv_batch_t, output_yolov2, max_prob_yolov2, detection_loss_yolov2, p_img_batch, printability_loss, patch_variation_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            # Calculate average loss over the course of training
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)
            #ep_yolov2_loss = ep_yolov2_loss/len(train_loader)
            #ep_yolov3_loss = ep_yolov3_loss/len(train_loader)
            ep_ssd_loss = ep_ssd_loss/len(train_loader)
            self.writer.add_scalar('loss/total_loss', ep_loss, epoch)
            self.writer.add_scalar('loss/total_det_loss', ep_det_loss, epoch)
            # self.writer.add_scalar('loss/YOLOv2_loss', ep_yolov2_loss, epoch)
            # self.writer.add_scalar('loss/YOLOv3_loss', ep_yolov3_loss, epoch)
            self.writer.add_scalar('loss/SSD_loss', ep_ssd_loss, epoch)
            self.writer.add_scalar('loss/printability_loss', ep_nps_loss, epoch)
            self.writer.add_scalar('loss/tv_loss',ep_tv_loss, epoch)

            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')
            # Output statistics and training time
            scheduler.step(ep_loss)
            print('  EPOCH NR: ', epoch),
            print('EPOCH LOSS: ', ep_loss)
            print('  DET LOSS: ', ep_det_loss)
            print('  NPS LOSS: ', ep_nps_loss)
            print('   TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1-et0)
            #del adv_batch_t, output_yolov2, max_prob_yolov2, detection_loss_yolov2, p_img_batch, printability_loss, patch_variation_loss, loss
            #torch.cuda.empty_cache()
            self.writer.add_image('patch', adv_patch.detach().cpu().numpy(), epoch)

            et0 = time.time()

        # At the end of training, save image
        im = transforms.ToPILImage('RGB')(adv_patch.cpu())
        plt.imshow(im)
        plt.show()
        # Specifies file to save trained patch to
        im.save("saved_patches/patch_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "-" + str(n_epochs) + "_epochs.jpg")

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


# Checks for the correct input length and then runs the trainer
def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()


if __name__ == '__main__':
    main()


