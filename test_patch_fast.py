import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json


if __name__ == '__main__':
    imgdir = "testing/clean"
    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolov2.weights"
    # To change the patch you're testing, change the patchfile variable to the path of the desired patch
    patchfile = "saved_patches/patch11.jpg"
    savedir = "testing"

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height

    patch_size = 300

    # Transform image to correct size
    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    # create tensor to represent image
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    clean_results = 0
    noise_results = 0
    patch_results = 0

    # Walk over clean images
    for imgfile in os.listdir(imgdir):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    # image name w/o extension
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))
            # open image and adjust to yolo input size
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w,h = img.size
            # read this label file back as a tensor
            if os.path.getsize(txtpath): # check to see if label file contains data.
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                # Eric: Unsure what purpose this unsqueeze serves
                label = label.unsqueeze(0)

            # Tensorify the image
            transform = transforms.ToTensor()
            img = transform(img).cuda()
            img_fake_batch = img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()
            
            # transform patch and add it to image
            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_p.png"
            p_img_pil.save(os.path.join(savedir, 'proper_patched/', properpatchedname))
            
            # check to see if the generated patch fools recognition
            boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
            boxes = nms(boxes, 0.4)
            for box in boxes:
                if box[6] == 0:
                    clean_results = clean_results + 1
                    if box[4].item() > 0.4: # if the threshold for detecting a person is met
                        patch_results = patch_results + 1

            # make a random patch, transform it and add it to the image
            random_patch = torch.rand(adv_patch_cpu.size()).cuda()
            adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_rdp.png"
            p_img_pil.save(os.path.join(savedir, 'random_patched/', properpatchedname))
            
            # check to see if the random patch evades recognition
            boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
            boxes = nms(boxes, 0.4)
            for box in boxes:
                if box[6] == 0 and box[4].item() > 0.4: # If the threshold for detecting a person is met
                        noise_results = noise_results + 1

    print("clean_results: " + str(clean_results/clean_results))
    print("noise_results: " + str(noise_results/clean_results))
    print("patch_results: " + str(patch_results/clean_results))
            

