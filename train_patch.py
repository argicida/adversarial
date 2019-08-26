"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time
import datetime


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        # TODO: note from Perry - eval() sets the target architecture to be static and not trainable,
        #  which is desired when training patches
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.non_printability_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        # Property in which most data is written to, including the patch
        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # Initialize some settings
        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 2000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        # Sets the patch tensor to have a gradient always, allowing the backward function
        adv_patch_cpu.requires_grad_(True)

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
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)

        # Schedules tasks on the gpu to optimize performance
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):

            # Sets the gradient inputs to zero
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()

            # I have no fucking clue how long this is supposed to be running, probably for the epoch length? Needs
            # More research
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():

                    # Optimizes everything to run on GPUs
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()

                    # Creates a patch transformer with a default grey patch. Can't find this object anywhere but most
                    # Likely it allows the patch to be moved around on inputted images.
                    # TODO: Find documentation for this object
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                    # Can't find this object anywhere else, most likely allows the patch to be put into inputted photos
                    # TODO: find documentation for this object
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))


                    # TODO: Figure out exactly what these transforms are doing
                    img = p_img_batch[1, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    # Looks to be where the given patch is evaluated, however all these objects are not included within
                    # The given darknet model. Documentation for the original needs to be found and researched
                    # TODO: Read documentation for original Darknet model
                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)

                    non_printability_score = self.non_printability_calculator(adv_patch)
                    patch_variation = self.total_variation(adv_patch)

                    # Calculates the loss in the new patch, then mashes them all together
                    printability_loss = non_printability_score*0.01
                    patch_variation_loss = patch_variation*2.5
                    detection_loss = torch.mean(max_prob)
                    loss = detection_loss + printability_loss + torch.max(patch_variation_loss, torch.tensor(0.1).cuda())
                    ep_det_loss += detection_loss.detach().cpu().numpy()
                    ep_nps_loss += printability_loss.detach().cpu().numpy()
                    ep_tv_loss += patch_variation_loss.detach().cpu().numpy()
                    ep_loss += loss

                    # Calculates the gradient of the loss function
                    loss.backward()

                    # Performs one step in optimization of the patch
                    optimizer.step()

                    # Clears all gradients after each step. Default is to accumulate them, we don't want that
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       # keep patch in image range

                    bt1 = time.time()
                    # Updates the iterations in batches of 5 and writes down new data
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        # Writes all this data to the object's tensorboard item, which was initialized as 'writer'
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', detection_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/printability_loss', printability_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', patch_variation_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        # Saves the current to argicida/patches
                        self.writer.add_image('patch', adv_patch_cpu, iteration)

                    # If the training is over, add an endline character, else clearn the following variables
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, detection_loss, p_img_batch, printability_loss, patch_variation_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            # Calculate average loss over the course of training
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')
            # Output statistics and training time
            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                del adv_batch_t, output, max_prob, detection_loss, p_img_batch, printability_loss, patch_variation_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

        # At the end of training, save image
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        plt.imshow(im)
        plt.show()
        # Specifies file to save trained patch to
        im.save("saved_patches/patch_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "-" + str(n_epochs) + "_epochs.jpg")

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

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


