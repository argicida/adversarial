import fnmatch
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir:str, max_lab:int, imgsize:int, target_names:list, shuffle=True, offset_labels=False):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = sum([len(fnmatch.filter(os.listdir(os.path.join(img_dir, "%s-labels"%target_name)), '*.txt'))
                        for target_name in target_names])
        assert n_images * len(target_names) == n_labels, "Number of images and number of labels don't match, %i, %i"\
                                                         %(n_images, n_labels)
        self.len = n_images
        self.img_dir = img_dir
        self.imgsize = imgsize
        img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.img_paths = [os.path.join(self.img_dir, img_name) for img_name in img_names]
        self.shuffle = shuffle
        self.target_lab_paths = {}
        for target_name in target_names:
            self.target_lab_paths[target_name] = []
        for target_name in target_names:
            lab_paths = [os.path.join(self.img_dir, "%s%s"%(target_name, "-labels"), img_name)
                             .replace('.jpg', '.txt').replace('.png', '.txt')
                         for img_name in img_names]
            self.target_lab_paths[target_name] = lab_paths
        self.max_n_labels = max_lab
        self.offset_labels = offset_labels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = self.img_paths[idx]
        label_by_target = {}
        image = Image.open(img_path).convert('RGB')
        for target_name in self.target_lab_paths:
            lab_path = self.target_lab_paths[target_name][idx]
            image = Image.open(img_path).convert('RGB')
            if os.path.getsize(lab_path):       #check to see if label file contains data.
                np_label = np.loadtxt(lab_path)
            else:
                np_label = np.ones([5])
            label = torch.from_numpy(np_label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)
            label_by_target[target_name] = self.pad_lab(label)
        if self.offset_labels:
            image, label_by_target = self.pad_and_scale(image, label_by_target)
        else:
            image, _ = self.pad_and_scale(image)
        transform = transforms.ToTensor()
        image = transform(image)
        return image, label_by_target

    def pad_and_scale(self, img:Image, label_by_target:dict=None):
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                if label_by_target:
                    for target_name in label_by_target:
                        lab = label_by_target[target_name]
                        lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                        lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                if label_by_target:
                    for target_name in label_by_target:
                        lab = label_by_target[target_name]
                        lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                        lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, label_by_target

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(lab, [0, 0, 0, pad_size], value=1)
        else:
            padded_lab = lab
        return padded_lab[0:self.max_n_labels]


class LegacyYolov2InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = self.img_paths[idx]
        lab_path = self.lab_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            np_label = np.loadtxt(lab_path)
        else:
            np_label = np.ones([5])

        label = torch.from_numpy(np_label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(lab, [0, 0, 0, pad_size], value=1)
        else:
            padded_lab = lab
        return padded_lab

