from typing import Callable
import os
import sys
import torch
from torchvision import transforms
from PIL import Image

from load_data import PatchTransformer, PatchApplier, InriaDataset
import numpy as np

from darknet import Darknet as Yolov2
from utils import do_detect as yolov2_detect

from implementations.yolov3.models import Darknet as Yolov3
from implementations.yolov3.utils import utils as yolov3_utils

from implementations.ssd.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor

import json
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import patches

VISUAL_DEBUG = False
PRINT_NMS_OUTPUT = False

LABELS_DIR = "./testing"

def main():
    print("Setting everything up")
    yolov2 = load_yolov2(0)
    ssd = load_ssd(0)
    yolov3 = load_yolov3(0)

    test_imgdir = "inria/Test/pos"
    # To change the patch you're testing, change the patchfile variable to the path of the desired patch
    patchfile = "saved_patches/perry_08-26_500_epochs.jpg"

    yolov2_img_size_sqrt = yolov2.height
    yolov3_img_size_sqrt = yolov2_img_size_sqrt
    ssd_img_size_sqrt = 300

    patch_size = 300

    # Transform image to correct size
    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    # create tensor to represent image
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    cols = ['target', 'image_size',
            'clean_num_obj', 'clean_num_human', 'clean_proparea_obj', 'clean_proparea_human',
            'noise_num_obj', 'noise_num_human', 'noise_proparea_obj', 'noise_proparea_human', 'noise_box_coverage_prop', 'noise_grand_iou',
            'patch_num_obj', 'patch_num_human', 'patch_proparea_obj', 'patch_proparea_human', 'patch_box_coverage_prop', 'patch_grand_iou'
           ]
    statistics = [['yolov2', yolov2_img_size_sqrt**2,
                   0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0],
                  ['yolov3', yolov3_img_size_sqrt**2,
                   0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0],
                  ['ssd', ssd_img_size_sqrt**2,
                   0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0]]
    statistics = pd.DataFrame(statistics, columns=cols)

    for imgfile in os.listdir(test_imgdir):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            image_name = os.path.splitext(imgfile)[0]    # image image_name w/o extension
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

            test_on_target(adv_patch, padded_img, image_name, yolov2, wrapper_yolov2,
                           'yolov2', yolov2_img_size_sqrt, statistics)
            test_on_target(adv_patch, padded_img, image_name, ssd, wrapper_ssd,
                           'ssd', ssd_img_size_sqrt, statistics)
            test_on_target(adv_patch, padded_img, image_name, yolov3, wrapper_yolov3,
                           'yolov3', yolov3_img_size_sqrt, statistics)
            if VISUAL_DEBUG:
                # show detections using pyplot state altered by detection functions
                plt.show()
    print("Done")
    print(statistics)

def test_on_target(adv_patch, padded_img, image_filename, target_function, target_wrapper_function,
                   target:str, input_size_sqrt:int, statistics:pd.DataFrame):
    tensorify = transforms.ToTensor()
    pilify = transforms.ToPILImage()

    resize_func = transforms.Resize(input_size_sqrt)
    resized_img = resize_func(padded_img)

    # it's possible to not run the clean detection every time, change in the future if necessary
    clean_boxes = target_wrapper_function(resized_img, target_function, input_size_sqrt, input_size_sqrt)
    save_architecture_ground_truths_if_none_exist(clean_boxes, input_size_sqrt, image_filename, target)

    ground_truths = load_ground_truths_as_tensor(image_filename, target).cuda()
    resized_img = tensorify(resized_img).cuda()
    resized_image_batched = resized_img.unsqueeze(0)
    ground_truths_batched = ground_truths.unsqueeze(0)
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    random_patch = torch.rand(adv_patch.size()).cuda()
    noise_transforms = patch_transformer(random_patch, ground_truths_batched, input_size_sqrt,
                                         do_rotate=True, rand_loc=True)
    noise_img_batched = patch_applier(resized_image_batched, noise_transforms)
    noise_img_unbatched_pil = transforms.ToPILImage('RGB')(noise_img_batched.squeeze(0).cpu())
    noise_boxes = target_wrapper_function(noise_img_unbatched_pil, target_function, input_size_sqrt, input_size_sqrt)

    patch_transforms = patch_transformer(adv_patch, ground_truths_batched, input_size_sqrt,
                                         do_rotate=True, rand_loc=True)
    patched_img_batched = patch_applier(resized_image_batched, patch_transforms)
    patched_img_unbatched_pil = pilify(patched_img_batched.squeeze(0).cpu())
    patched_boxes = target_wrapper_function(patched_img_unbatched_pil, target_function, input_size_sqrt, input_size_sqrt)

    update_single_image_box_statistics(statistics, clean_boxes, target, 'clean',
                                       resized_img.cpu().transpose(0, 2).transpose(0, 1))
    update_single_image_box_statistics(statistics, noise_boxes, target, 'noise',
                                       np.array(noise_img_unbatched_pil))
    update_single_image_box_statistics(statistics, patched_boxes, target, 'patch',
                                       np.array(patched_img_unbatched_pil))
    update_patched_images_statistics(statistics, clean_boxes, noise_boxes, patched_boxes, target)
    return


def display_bounding_boxes(input_image:Image.Image, x0_y0_width_height_human_dataframe, title) -> None:
    fig, ax = plt.subplots(1)
    ax.imshow(input_image)
    for index, box in x0_y0_width_height_human_dataframe.iterrows():
        color = 'r' if box['human'] else 'g'
        rect = patches.Rectangle((box['x0'], box['y0']), box['w'], box['h'], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    plt.title(title)
    return


def update_single_image_box_statistics(statistics:pd.DataFrame, x0_y0_width_height_human_dataframe:pd.DataFrame,
                                       target:str, img_type:str, input_image=None) -> None:
    if VISUAL_DEBUG:
        assert input_image is not None
        display_bounding_boxes(input_image, x0_y0_width_height_human_dataframe, target)
    assert img_type == 'clean' or img_type == 'noise' or img_type == 'patch'
    row_mask = (statistics['target'] == target)
    image_size = statistics.loc[row_mask, 'image_size']
    num_obj = 0
    num_human = 0
    area_obj = 0
    area_human = 0
    for index, box in x0_y0_width_height_human_dataframe.iterrows():
        area = box['w'] * box['h'] / image_size
        num_obj += 1
        area_obj += area
        if box['human']:
            num_human += 1
            area_human += area
    statistics.loc[row_mask, img_type + '_num_obj'] += num_obj
    statistics.loc[row_mask, img_type + '_num_human'] += num_human
    statistics.loc[row_mask, img_type + '_proparea_obj'] += area_obj
    statistics.loc[row_mask, img_type + '_proparea_human'] += area_human
    return

def bitwise_boxes_area_union(x0_y0_width_height_human_dataframe:pd.DataFrame, image_size:int) -> np.ndarray:
    side_length = int(image_size**0.5)
    area = np.zeros((side_length, side_length), dtype=bool)
    for index, box in x0_y0_width_height_human_dataframe.iterrows():
        mask = np.zeros((side_length, side_length), dtype=bool)
        x0 = box['x0']
        y0 = box['y0']
        w = box['w']
        h = box['h']
        mask[int(x0):int(x0+w), int(y0):int(y0+h)] = True
        area = np.logical_or(area, mask)
    return area


def generate_label_filepath(target:str, image_name:str) -> str:
    return os.path.abspath(os.path.join(LABELS_DIR,'clean/', target + '-labels/', image_name + '.txt'))


def save_architecture_ground_truths_if_none_exist(x0_y0_width_height_human_dataframe_clean:pd.DataFrame,
                                                  image_size_sqrt:int, image_name:str, target:str) -> None:
    label_filepath = generate_label_filepath(target, image_name)
    try:
        _ = os.path.getsize(label_filepath)
    except FileNotFoundError:
        directory = os.path.dirname(label_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        textfile = open(label_filepath, 'w+')
        side_length = image_size_sqrt
        for index, box in x0_y0_width_height_human_dataframe_clean.iterrows():
            norm_x0 = box['x0'] / side_length
            norm_y0 = box['y0'] / side_length
            norm_width = box['w'] / side_length
            norm_height = box['h'] / side_length
            if box['human']:
                cls_id = 0 # for legacy code compatibility
                textfile.write(f'{cls_id} {norm_x0 + 0.5*norm_width} {norm_y0 + 0.5*norm_height} {norm_width} {norm_height}\n')
        textfile.close()
    return


def load_ground_truths_as_tensor(image_name:str, target:str) -> torch.Tensor:
    label_filepath = generate_label_filepath(target, image_name)
    label = None
    try:
        if os.path.getsize(label_filepath):
            label = np.loadtxt(label_filepath)
        else:
            # stub for when there are no human detected in ground truth (clean images)
            label = np.ones([5])
    except FileNotFoundError:
        print('label %s not found'%label_filepath, file=sys.stderr)
        exit(1)
    label = torch.from_numpy(label).float()
    if label.dim() == 1:
        # assume multiple boxes in an image
        label = label.unsqueeze(0)
    return label


def update_patched_images_statistics(statistics:pd.DataFrame,
                                     x0_y0_width_height_human_dataframe_clean:pd.DataFrame,
                                     x0_y0_width_height_human_dataframe_noise:pd.DataFrame,
                                     x0_y0_width_height_human_dataframe_patch:pd.DataFrame,
                                     target:str) -> None:
    row_mask = (statistics['target'] == target)
    image_size = statistics.loc[row_mask, 'image_size']
    clean_box_area = bitwise_boxes_area_union(x0_y0_width_height_human_dataframe_clean, image_size)
    noise_box_area = bitwise_boxes_area_union(x0_y0_width_height_human_dataframe_noise, image_size)
    patch_box_area = bitwise_boxes_area_union(x0_y0_width_height_human_dataframe_patch, image_size)
    covered_by_noise = np.logical_and(clean_box_area, noise_box_area)
    covered_by_patch = np.logical_and(clean_box_area, patch_box_area)
    noise_coverage_proportion = np.count_nonzero(covered_by_noise) / image_size
    patch_coverage_proportion = np.count_nonzero(covered_by_patch) / image_size
    noise_grand_iou = np.count_nonzero(covered_by_noise) / np.count_nonzero(np.logical_or(clean_box_area, noise_box_area))
    patch_grand_iou = np.count_nonzero(covered_by_patch) / np.count_nonzero(np.logical_or(clean_box_area, patch_box_area))
    statistics.loc[row_mask, 'noise_box_coverage_prop'] += noise_coverage_proportion
    statistics.loc[row_mask, 'patch_box_coverage_prop'] += patch_coverage_proportion
    statistics.loc[row_mask, 'noise_grand_iou'] += noise_grand_iou
    statistics.loc[row_mask, 'patch_grand_iou'] += patch_grand_iou
    return


def wrapper_yolov3(image, net, input_w, input_h) -> pd.DataFrame:
    detection_confidence_threshold = 0.5
    nms_thres = 0.4

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
    info = []
    if boxes is not None:
        for box in boxes:
            x0, y0, x1, y1, conf, cls_conf, cls = box
            human = False
            x0 = x0.detach().cpu().numpy()
            y0 = y0.detach().cpu().numpy()
            x1 = x1.detach().cpu().numpy()
            y1 = y1.detach().cpu().numpy()
            if PRINT_NMS_OUTPUT:
                print("YOLOV3 %s"%(str(box)))
            if cls == 0:
                human = True
            info.append([x0, y0, x1-x0, y1-y0, human])
    return pd.DataFrame(info, columns=["x0", "y0", "w", 'h', 'human'])


def wrapper_ssd(image, net, input_w, input_h) -> pd.DataFrame:
    class_person = 15
    tensor = None
    if isinstance(image, Image.Image):
        width = image.width
        height = image.height
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = tensor.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous().cuda()
        tensor = tensor.view(1, 3, height, width)
        tensor = tensor.float().cuda()
    elif type(image) == np.ndarray:  # cv2 image
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    else:
        print("unknown image type")
        exit(-1)
    tensor[:, 0, :, :] -= 123
    tensor[:, 1, :, :] -= 117
    tensor[:, 2, :, :] -= 104
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        boxes, labels, probs = net(tensor)
    info = []
    for i, output_class in enumerate(labels):
        box = boxes[i]
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        x0 = x0.detach().cpu().numpy()
        y0 = y0.detach().cpu().numpy()
        x1 = x1.detach().cpu().numpy()
        y1 = y1.detach().cpu().numpy()
        human = False
        if PRINT_NMS_OUTPUT:
            print("SSD %s" % (str(boxes[i])))
        if output_class == class_person:
            human = True
        info.append([x0, y0, x1 - x0, y1 - y0, human])
    return pd.DataFrame(info, columns=["x0", "y0", "w", 'h', 'human'])


def wrapper_yolov2(image, darknet_model, input_w, input_h) -> pd.DataFrame:
    detection_confidence_threshold = 0.5
    nms_threshold = 0.4
    with torch.no_grad():
        boxes = yolov2_detect(darknet_model, image, detection_confidence_threshold, nms_threshold, use_cuda=True)
    info = []
    for box in boxes:
        if PRINT_NMS_OUTPUT:
            print("YOLOv2 %s" % (str(box)))
        center_x = box[0]
        center_y = box[1]
        normalized_w = box[2]
        normalized_h = box[3]
        x0 = (center_x - normalized_w / 2.0) * input_w
        y0 = (center_y - normalized_h / 2.0) * input_h
        width = normalized_w * input_w
        height = normalized_h * input_h

        width = width.detach().cpu().numpy()
        height = height.detach().cpu().numpy()
        x0 = x0.detach().cpu().numpy()
        y0 = y0.detach().cpu().numpy()

        human = False
        if box[6] == 0:
            human = True
        info.append([x0, y0, width, height, human])
    return pd.DataFrame(info, columns=["x0", "y0", "w", 'h', 'human'])


def load_yolov2(device) -> torch.nn.Module:
    yolov2_cfgfile = "cfg/yolov2.cfg"
    yolov2_weightfile = "weights/yolov2.weights"
    yolov2 = Yolov2(yolov2_cfgfile)
    yolov2.load_weights(yolov2_weightfile)
    return yolov2.eval().cuda(device)


def load_yolov3(device) -> torch.nn.Module:
    yolov3_cfgfile = "./implementations/yolov3/config/yolov3.cfg"
    yolov3_weightfile = "./implementations/yolov3/weights/yolov3.weights"
    yolov3 = Yolov3(yolov3_cfgfile)
    yolov3.load_darknet_weights(yolov3_weightfile)
    return yolov3.cuda(device)


def load_ssd(device) -> Callable:
    ssd_weightfile = "./implementations/ssd/models/vgg16-ssd-mp-0_7726.pth"
    ssd = create_vgg_ssd(21, is_test=True)
    ssd.load(ssd_weightfile)
    ssd = ssd.cuda(device)
    single_image_predictor = create_vgg_ssd_predictor(ssd, nms_method="hard", device=device)
    predict_function = single_image_predictor.predict
    return predict_function


if __name__ == '__main__':
    main()
