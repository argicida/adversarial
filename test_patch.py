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
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

from implementations.ssd_pytorch.ssd import build_ssd
from implementations.ssd_pytorch.data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from implementations.ssd_pytorch.data import VOC_CLASSES as labels
import cv2
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from implementations.yolov3.models import Darknet as Yolov3


def test_results_ssd(image, net):
    human_positives = 0
    total_positives = 0
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (300, 300))
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(image.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    detections = y.data
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            if labels[i-1] == "person":
                human_positives += 1
            total_positives += 1
    return human_positives, total_positives


def test_results_darknet(image, darknet_model):
    detection_confidence_threshold = 0.5
    nms_threshold = 0.4
    human_box_count = 0
    boxes = do_detect(darknet_model, image, detection_confidence_threshold, nms_threshold, use_cuda=True)
    # boxes = nms(boxes, nms_threshold)
    for box in boxes:
        if box[6] == 0:
            human_box_count = human_box_count + 1
    total_box_count = len(boxes)
    return human_box_count, total_box_count


def load_yolov2():
    yolov2_cfgfile = "cfg/yolov2.cfg"
    yolov2_weightfile = "weights/yolov2.weights"
    yolov2 = Darknet(yolov2_cfgfile)
    yolov2.load_weights(yolov2_weightfile)
    return yolov2.eval().cuda()


def load_yolov3():
    yolov3_cfgfile = ""
    yolov3_weightfile = ""
    yolov3 = Yolov3(yolov3_cfgfile)
    yolov3.load_darknet_weights(yolov3_weightfile)
    return yolov3.eval().cuda()


if __name__ == '__main__':
    # print("Setting everything up")
    yolov2 = load_yolov2()

    ssd_model = build_ssd('test', 300, 21)    # initialize ssd
    ssd_model.load_weights('./implementations/ssd_pytorch/weights/ssd300_mAP_77.43_v2.pth')

    yolov3 = load_yolov3()

    test_imgdir = "inria/Test/pos"
    cachedir = "testing"
    # To change the patch you're testing, change the patchfile variable to the path of the desired patch
    patchfile = "saved_patches/perry_09-14_19-46-08-1000_epochs.jpg"

    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    batch_size = 1
    max_lab = 14
    img_size = yolov2.height

    patch_size = 300

    # Transform image to correct size
    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    # create tensor to represent image
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    # clean_results = []
    yolov2_clean_human_positives = 0
    yolov2_clean_object_positives = 0
    ssd_clean_human_positives = 0
    ssd_clean_object_positives = 0
    # noise_results = []
    yolov2_noise_human_positives = 0
    yolov2_noise_object_positives = 0
    ssd_noise_human_positives = 0
    ssd_noise_object_positives = 0
    # patch_results = []
    yolov2_patch_human_positives = 0
    yolov2_patch_object_positives = 0
    ssd_patch_human_positives = 0
    ssd_patch_object_positives = 0

    # Walk over clean images
    for imgfile in os.listdir(test_imgdir):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    # image name w/o extension
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(cachedir, 'clean/', 'yolo-labels/', txtname))
            # open image and adjust to yolo input size
            imgfile = os.path.abspath(os.path.join(test_imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w,h = img.size
            # ensure image is square
            if w==h:
                padded_img = img
            else:
                # pad image with grey
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127, 127, 127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                    padded_img.paste(img, (0, int(padding)))
            # resize image to fit into yolo neural net
            resize = transforms.Resize((img_size, img_size))
            padded_img = resize(padded_img)
            cleanname = name + ".png"
            # save this image
            # padded_img.save(os.path.join(savedir, 'clean/', cleanname))

            """ at this point, clean images are prepped to be analyzed by yolo """
            human_positives, object_positives = test_results_darknet(padded_img, yolov2)
            yolov2_clean_human_positives += human_positives
            yolov2_clean_object_positives += object_positives
            human_positives, object_positives = test_results_ssd(cleanname, ssd_model)
            ssd_clean_human_positives += human_positives
            ssd_clean_object_positives += object_positives
            '''
            # generate a label file for the padded image
            boxes = do_detect(darknet_model, padded_img, 0.5, 0.4, True) # run yolo object detection on image
            #boxes = nms(boxes, 0.4) # run non-maximum suppression to remove redundant boxes, already called in do_detect()
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   # if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    # add detected box to label file (only for people)
                    clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                     y_center.item() - height.item() / 2,
                                                                     width.item(),
                                                                     height.item()],
                                          'score': box[4].item(),
                                          'category_id': 1})
            textfile.close()
            '''

            """ At this point, image recognition has been ran, and humans detected in images have been tracked"""

            # read this label file back as a tensor
            if os.path.getsize(txtpath):       #check to see if label file contains data.
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                # Eric: Unsure what purpose this unsqueeze serves
                # Perry: it adds an extra dimension to work with batched/vectorized algorithms,
                #   which are designed to take in multiple labels at once
                label = label.unsqueeze(0)

            # Tensorify the image
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()

            # transform patch and add it to image
            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_p.png"
            # p_img_pil.save(os.path.join(savedir, 'proper_patched/', properpatchedname))

            # generate a label file for the image with sticker
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(cachedir, 'proper_patched/', 'yolo-labels/', txtname))
            human_positives, object_positives = test_results_darknet(p_img_pil, yolov2)
            yolov2_patch_human_positives += human_positives
            yolov2_patch_object_positives += object_positives
            human_positives, object_positives = test_results_ssd(p_img_pil)
            ssd_patch_human_positives += human_positives
            ssd_patch_object_positives += object_positives
            '''
            boxes = do_detect(darknet_model, p_img_pil, 0.5, 0.4, True)
            #boxes = nms(boxes, 0.4)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   # if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()
            '''

            # make a random patch, transform it and add it to the image
            random_patch = torch.rand(adv_patch_cpu.size()).cuda()
            adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_rdp.png"
            # p_img_pil.save(os.path.join(savedir, 'random_patched/', properpatchedname))

             # generate a label file for the random patch image
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(cachedir, 'random_patched/', 'yolo-labels/', txtname))
            human_positives, object_positives = test_results_darknet(p_img_pil, yolov2)
            yolov2_noise_human_positives += human_positives
            yolov2_noise_object_positives += object_positives
            human_positives, object_positives = test_results_ssd(p_img_pil)
            ssd_noise_human_positives += human_positives
            ssd_noise_object_positives += object_positives
            '''
            boxes = do_detect(darknet_model, p_img_pil, 0.5, 0.4, True)
            #boxes = nms(boxes, 0.4)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   # if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()
    print(total)
    with open('clean_results.json', 'w') as fp:
        json.dump(clean_results, fp)
    with open('noise_results.json', 'w') as fp:
        json.dump(noise_results, fp)
    with open('patch_results.json', 'w') as fp:
        json.dump(patch_results, fp)
        
    '''
    print("Done")
    results = open('test_results.txt', 'w+')
    results.write('yolov2 results\n')
    results.write(f'noise to clean human positive ratio: {yolov2_noise_human_positives / yolov2_clean_human_positives}\n')
    results.write(f'patch to clean human positive ratio: {yolov2_patch_human_positives / yolov2_clean_human_positives}\n')
    results.write(f'noise to clean object positive ratio: {yolov2_noise_object_positives / yolov2_clean_object_positives}\n')
    results.write(f'patch to clean object positive ratio: {yolov2_patch_object_positives / yolov2_clean_object_positives}\n')
    results.write('ssd results\n')
    results.write(f'noise to clean human positive ratio: {ssd_noise_human_positives / ssd_clean_human_positives}\n')
    results.write(f'patch to clean human positive ratio: {ssd_patch_human_positives / ssd_clean_human_positives}\n')
    results.write(f'noise to clean object positive ratio: {ssd_noise_object_positives / ssd_clean_object_positives}\n')
    results.write(f'patch to clean object positive ratio: {ssd_patch_object_positives / ssd_clean_object_positives}\n')
    results.close()
    # stats = open('test_results.csv', 'a+')
    # stats.write(f'{noise_object_positives / clean_object_positives},{noise_human_positives / clean_human_positives},'
    # f'{patch_object_positives / clean_object_positives},{patch_human_positives / clean_human_positives}\n')
    # stats.close()
