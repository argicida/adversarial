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

from implementations.yolov3.models import Darknet as Yolov3
from implementations.yolov3.utils import utils as yolov3_utils

from implementations.ssd.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor

def test_results_yolov3(image, net):
    detection_confidence_threshold = 0.5
    nms_thres = 0.4
    human_positives = 0
    total_positives = 0

    tensor = None
    if isinstance(image, Image.Image):
        width = image.width
        height = image.height
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = tensor.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous().cuda()
        tensor = tensor.view(1, 3, height, width)
        tensor = tensor.float().div(255.0)
    elif type(image) == np.ndarray:  # cv2 image
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).cuda()
    else:
        print("unknown image type")
        exit(-1)

    with torch.no_grad():
        outputs = net(tensor)
        outputs = yolov3_utils.non_max_suppression(outputs, conf_thres=detection_confidence_threshold, nms_thres=nms_thres)
        boxes = outputs[0]
    if boxes is not None:
        for box in boxes:
            if box[6] == 0:
                human_positives += 1
        total_positives = len(boxes)
    return human_positives, total_positives


def test_results_ssd(image, net):
    human_positives = 0
    total_positives = 0
    tensor = None
    if isinstance(image, Image.Image):
        width = image.width
        height = image.height
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = tensor.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous().cuda()
        tensor = tensor.view(1, 3, height, width)
        tensor = tensor.float().div(255.0).cuda()
    elif type(image) == np.ndarray:  # cv2 image
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).cuda()
    else:
        print("unknown image type")
        exit(-1)
    tensor[:, 0, :, :] -= 123
    tensor[:, 1, :, :] -= 117
    tensor[:, 2, :, :] -= 104
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    boxes, labels, probs = net(tensor)
    for i in range(len(labels)):
        clas = labels[i]
        if clas == 15:
            human_positives += 1
    total_positives = len(labels)
    return human_positives, total_positives


def test_results_yolov2(image, darknet_model):
    detection_confidence_threshold = 0.5
    nms_threshold = 0.4
    human_box_count = 0
    with torch.no_grad():
        boxes = do_detect(darknet_model, image, detection_confidence_threshold, nms_threshold, use_cuda=True)
    # boxes = nms(boxes, nms_threshold)
    for box in boxes:
        if box[6] == 0:
            human_box_count = human_box_count + 1
    total_box_count = len(boxes)
    return human_box_count, total_box_count


def load_yolov2(device):
    yolov2_cfgfile = "cfg/yolov2.cfg"
    yolov2_weightfile = "weights/yolov2.weights"
    yolov2 = Darknet(yolov2_cfgfile)
    yolov2.load_weights(yolov2_weightfile)
    return yolov2.eval().cuda(device)


def load_yolov3(device):
    yolov3_cfgfile = "./implementations/yolov3/config/yolov3.cfg"
    yolov3_weightfile = "./implementations/yolov3/weights/yolov3.weights"
    yolov3 = Yolov3(yolov3_cfgfile)
    yolov3.load_darknet_weights(yolov3_weightfile)
    return yolov3.cuda(device)


def load_ssd(device):
    ssd_weightfile = "./implementations/ssd/models/vgg16-ssd-mp-0_7726.pth"
    ssd = create_vgg_ssd(21, is_test=True)
    ssd.load(ssd_weightfile)
    ssd = ssd.cuda(device)
    single_image_predictor = create_vgg_ssd_predictor(ssd, device=device)
    predict_function = single_image_predictor.predict
    return predict_function


def main():
    print("Setting everything up")
    yolov2 = load_yolov2(0)
    ssd = load_ssd(0)
    yolov3 = load_yolov3(0)

    test_imgdir = "inria/Test/pos"
    cachedir = "testing"
    # To change the patch you're testing, change the patchfile variable to the path of the desired patch
    patchfile = "saved_patches/patch_2019-10-14_13-52-43-2000_epochs.jpg"

    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    batch_size = 1
    max_lab = 14
    img_size = yolov2.height
    ssd_img_size = 300

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
    yolov3_clean_human_positives = 0
    yolov3_clean_object_positives = 0
    # noise_results = []
    yolov2_noise_human_positives = 0
    yolov2_noise_object_positives = 0
    ssd_noise_human_positives = 0
    ssd_noise_object_positives = 0
    yolov3_noise_human_positives = 0
    yolov3_noise_object_positives = 0
    # patch_results = []
    yolov2_patch_human_positives = 0
    yolov2_patch_object_positives = 0
    ssd_patch_human_positives = 0
    ssd_patch_object_positives = 0
    yolov3_patch_human_positives = 0
    yolov3_patch_object_positives = 0

    print("Walking over clean images")
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
            # resize image to fit the ssd neural net
            ssd_resize = transforms.Resize((ssd_img_size, ssd_img_size))
            ssd_padded_img = ssd_resize(padded_img)
            padded_img = resize(padded_img)
            cleanname = name + ".png"
            # save this image
            # padded_img.save(os.path.join(savedir, 'clean/', cleanname))

            """ at this point, clean images are prepped to be analyzed by yolo """
            print("Ready to be analyzed")
            human_positives, object_positives = test_results_yolov2(padded_img, yolov2)
            yolov2_clean_human_positives += human_positives
            yolov2_clean_object_positives += object_positives

            human_positives, object_positives = test_results_ssd(ssd_padded_img, ssd)
            ssd_clean_human_positives += human_positives
            ssd_clean_object_positives += object_positives

            human_positives, object_positives = test_results_yolov3(padded_img, yolov3)
            yolov3_clean_human_positives += human_positives
            yolov3_clean_object_positives += object_positives

            try:
                _ = os.path.getsize(txtpath)
            except FileNotFoundError:
                # generate a label file for the padded image
                boxes = do_detect(yolov2, padded_img, 0.5, 0.4, True) # run yolo object detection on image
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
                textfile.close()
            """
            generate a label file for the padded image
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
            """

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
            human_positives, object_positives = test_results_yolov2(p_img_pil, yolov2)
            yolov2_patch_human_positives += human_positives
            yolov2_patch_object_positives += object_positives
            
            ssd_patched_img_pillow = ssd_resize(p_img_pil)
            human_positives, object_positives = test_results_ssd(ssd_patched_img_pillow, ssd)
            ssd_patch_human_positives += human_positives
            ssd_patch_object_positives += object_positives
            
            human_positives, object_positives = test_results_yolov3(p_img_pil, yolov3)
            yolov3_patch_human_positives += human_positives
            yolov3_patch_object_positives += object_positives
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
            human_positives, object_positives = test_results_yolov2(p_img_pil, yolov2)
            yolov2_noise_human_positives += human_positives
            yolov2_noise_object_positives += object_positives
            
            ssd_patched_img_pillow = ssd_resize(p_img_pil)
            human_positives, object_positives = test_results_ssd(ssd_patched_img_pillow, ssd)
            ssd_noise_human_positives += human_positives
            ssd_noise_object_positives += object_positives
            
            human_positives, object_positives = test_results_yolov3(p_img_pil, yolov3)
            yolov3_noise_human_positives += human_positives
            yolov3_noise_object_positives += object_positives
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
    results.write(patchfile)
    results.write('\nyolov2 results\n')
    results.write(f'noise to clean human positive ratio: {yolov2_noise_human_positives / yolov2_clean_human_positives}\n')
    results.write(f'patch to clean human positive ratio: {yolov2_patch_human_positives / yolov2_clean_human_positives}\n')
    results.write(f'noise to clean object positive ratio: {yolov2_noise_object_positives / yolov2_clean_object_positives}\n')
    results.write(f'patch to clean object positive ratio: {yolov2_patch_object_positives / yolov2_clean_object_positives}\n')
    results.write('ssd results\n')
    results.write(f'noise to clean human positive ratio: {ssd_noise_human_positives / ssd_clean_human_positives}\n')
    results.write(f'patch to clean human positive ratio: {ssd_patch_human_positives / ssd_clean_human_positives}\n')
    results.write(f'noise to clean object positive ratio: {ssd_noise_object_positives / ssd_clean_object_positives}\n')
    results.write(f'patch to clean object positive ratio: {ssd_patch_object_positives / ssd_clean_object_positives}\n')
    results.write('yolov3 results\n')
    results.write(f'noise to clean human positive ratio: {yolov3_noise_human_positives / yolov3_clean_human_positives}\n')
    results.write(f'patch to clean human positive ratio: {yolov3_patch_human_positives / yolov3_clean_human_positives}\n')
    results.write(f'noise to clean object positive ratio: {yolov3_noise_object_positives / yolov3_clean_object_positives}\n')
    results.write(f'patch to clean object positive ratio: {yolov3_patch_object_positives / yolov3_clean_object_positives}\n')
    results.close()
    # stats = open('test_results.csv', 'a+')
    # stats.write(f'{noise_object_positives / clean_object_positives},{noise_human_positives / clean_human_positives},'
    # f'{patch_object_positives / clean_object_positives},{patch_human_positives / clean_human_positives}\n')
    # stats.close()


if __name__ == '__main__':
    main()
