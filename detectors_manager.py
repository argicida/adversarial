import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import torch.nn.functional as F

# yolov2
from darknet import Darknet
# yolov3
from implementations.yolov3.models import Darknet as Yolov3
# ssd
from implementations.ssd.vision.ssd.vgg_ssd import create_vgg_ssd
from implementations.ssd.vision.utils.box_utils import convert_locations_to_boxes

from patch_utilities import SquarePatchTransformApplier

from absl import flags

import cli_config # defines the flags, dont remove this line
FLAGS = flags.FLAGS

# some of these have multiple output types, such as class confidence and object confidence, so multiple setting
SUPPORTED_TRAIN_DETECTORS = {'yolov2':3, # 1: class only, 2: object only, 3: object and class
                             'ssd':1, # 1: class
                             'yolov3':3 # 1: class only, 2: object only, 3: object and class
                            }


class Manager:
  def __init__(self, detector_device_mapping:dict, train_detector_settings:dict, activate_logits:bool=False,
               debug_autograd:bool=False, debug_device:bool=False, debug_coords:bool=False, plot_patches:bool=False,
               test_detectors:list=None): # testing currently not supported
    # all detectors in train_detector_settings and test_detectors must be mapped to a device in detector_device_mapping
    self.train_detector_settings = train_detector_settings
    self.detector_instances = {}
    self.output_process_fns = {}
    self.patch_appliers_by_device = {}
    self.debug_autograd = debug_autograd
    self.debug_device = debug_device
    self.debug_coords = debug_coords
    self.plot_patches = plot_patches
    for detector_name, cuda_device_id in detector_device_mapping.items():
      target_setting = train_detector_settings[detector_name]
      detector = _create_detector(detector_name, device=cuda_device_id, activate_logits=activate_logits)
      self.detector_instances[detector_name] = detector
      if cuda_device_id not in self.patch_appliers_by_device:
        self.patch_appliers_by_device[cuda_device_id] = SquarePatchTransformApplier(cuda_device_id)
      if target_setting == 1:
        self.output_process_fns[detector_name] = [detector.raw_output_to_people_conf_cxcy]
      elif target_setting == 2:
        self.output_process_fns[detector_name] = [detector.raw_output_to_objects_conf_cxcy]
      elif target_setting == 3:
        self.output_process_fns[detector_name] = [detector.raw_output_to_people_conf_cxcy,
                                                  detector.raw_output_to_objects_conf_cxcy]
      else:
        assert False, "Invalid Detector Setting: %s %i"%(detector_name, target_setting)

  def _conf_grad_ok(self, confidences:torch.Tensor):
    return confidences.requires_grad and (confidences.grad_fn is not None)

  def _plot_patched_images_comparison(self, input_images:torch.Tensor, patched_images:torch.Tensor,
                                      boxes:torch.Tensor, detector_name:str):
    num_images = input_images.shape[0]
    height = input_images.shape[2]
    width = input_images.shape[3]
    transpose = lambda x: np.transpose(x, axes=[0, 2, 3, 1])
    clean_images = transpose((input_images.detach().cpu().numpy() * 255).astype('uint8'))
    patched_images = transpose((patched_images.detach().cpu().numpy() * 255).astype('uint8'))
    boxes = boxes.detach().cpu().numpy()
    fig, axes = plt.subplots(2, num_images)
    for index, clean_img in enumerate(clean_images):
      patched_img = patched_images[index]
      clean_img_pil = Image.fromarray(clean_img)
      patched_img_pil = Image.fromarray(patched_img)
      axes[0, index].imshow(clean_img_pil)
      axes[1, index].imshow(patched_img_pil)
      for c_x_y_w_h in boxes[index]:
        c, x, y, w, h = c_x_y_w_h
        color = 'r' if c==0 else 'g'
        x = x * width
        w = w * width
        y = y * height
        h = h * height
        rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
        axes[1, index].add_patch(rect)
    fig.suptitle(detector_name)
    plt.show()

  def train_forward_propagate(self, input_images:torch.Tensor, labels_by_target:dict, patch_2d:torch.Tensor=None) \
        -> dict:
    # returns a dict of results, containing one or more logit maps for each target depending on setting
    # , as well as a copy a label for each target on their respective gpu device
    # pass in None for patch parameter if input_images are rendered from 3D model
    patching_enabled = (patch_2d is not None)
    output_maps = {}
    copies_of_inputs_by_device = {}
    for target_name, target_setting in self.train_detector_settings.items():
      target = self.detector_instances[target_name]
      labels = labels_by_target[target_name]
      target_cuda_device = torch.device('cuda:%i' % target.device)
      if target.device not in copies_of_inputs_by_device:
        patch_copy = patch_2d.cuda(device=target_cuda_device, non_blocking=True) if patching_enabled else None
        images_copy = input_images.cuda(device=target_cuda_device, non_blocking=True)
        copies_of_inputs_by_device[target.device] = (images_copy, patch_copy)
      else:
        images_copy, patch_copy = copies_of_inputs_by_device[target.device]
      labels_copy = labels.cuda(device=target_cuda_device, non_blocking=True)
      detector_input = self.patch_appliers_by_device[target.device](patch_copy, images_copy, labels_copy) \
          if patching_enabled else images_copy
      if self.plot_patches: self._plot_patched_images_comparison(images_copy, detector_input, labels_copy, target_name)
      raw_output = target.batched_forward_to_raw_output(detector_input)
      #labels_copy = target.normalize_labels(labels_copy)

      outputs = []
      for fn in self.output_process_fns[target_name]:
        conf, cxcy = fn(raw_output)
        assert (not self.debug_autograd) or self._conf_grad_ok(conf), "%s gradients"%target_name
        assert (not self.debug_device) or ((conf.device == target_cuda_device) and (cxcy.device == target_cuda_device)),\
            "%s output device inconsistent"%target_name
        assert (not self.debug_coords) or (torch.max(cxcy) < 2), \
            "%s box locations not normalized: %.2f"%(target_name, float(torch.max(cxcy)))
        outputs.append((conf, cxcy, labels_copy))
      output_maps[target_name] = outputs
    return output_maps

  def single_image_detection_on_targets(self, square_padded_img:Image) -> dict:
    # returns dict of str:np.ndarray,
    # with str being target name and array being tuples of (is_person, x0, y0, w, h)
    # output = {}
    # with torch.no_grad():
    #   for target in self.test_detectors:
    #     output[target] = self.detector_instances[target].single_image_nms_predictions(square_padded_img)
    # return output
    # used for testing in the future, for now use existing code in test_patch.py
    raise NotImplementedError


class _AbstractDetector:
  def __init__(self, device:int, activate_logits:bool, input_w:int, input_h:int):
    self.device = device
    self.model = self._load_model(device)
    self.activate_logits = activate_logits
    self.input_w = input_w
    self.input_h = input_h
    return

  def batched_forward_to_raw_output(self, batched_images:torch.Tensor):
    """
    :param batched_images: batched images on the same device as the detector
    :return: depends on the detector
    """
    return self.model(self._batched_preprocess(batched_images))

  def _load_model(self, device:int) -> torch.nn.Module:
    """
    :param device: cuda device id
    :return: an nn.module that forward pass batched images into a tensor of logits
    """
    raise NotImplementedError

  def _batched_preprocess(self, batched_images: torch.Tensor) -> torch.Tensor:
    """
    :param batched_images: batched images on the same device as the detector
    :return: normalized, zero meaned, etc. whatever needed for the particular detector
    """
    raise NotImplementedError

  def raw_output_to_objects_conf_cxcy(self, raw_output):
    """
    :param raw_output: raw output from batched_forward_to_raw_output
    :return: object confidence logits/scores [batched size, num_predictions],
        cx_cy [batched size, num_predictions, 2] normalized
    """
    raise NotImplementedError

  def raw_output_to_people_conf_cxcy(self, raw_output):
    """
    :param raw_output: raw output from batched_forward_to_raw_output
    :return: person confidence logits/scores [batched size, num_predictions],
        cx_cy [batched size, num_predictions, 2] normalized
    """
    raise NotImplementedError

  def normalize_labels(self, labels: torch.Tensor):
    """
    :param labels: [batch size, num_labels, 5] gt boxes for person bounding boxes consisted of
        (is_person, x_0, y_0, w, h), should be scaled to detector input size
    :return: [batch size, num_labels, 5] gt boxes for person bounding boxes consisted of
        (is_person, x_0, y_0, w, h), should be scaled to (0, 1)
    """
    labels_clone = labels.clone()
    labels_clone[:, :, 1] = labels_clone[:, :, 1] / self.input_w
    labels_clone[:, :, 3] = labels_clone[:, :, 3] / self.input_w
    labels_clone[:, :, 2] = labels_clone[:, :, 2] / self.input_h
    labels_clone[:, :, 4] = labels_clone[:, :, 4] / self.input_h
    return labels_clone

  def single_image_nms_predictions(self, single_square_image:Image) -> np.ndarray:
    # used for testing, which goes through images one by one
    # returns array of normalized (is_person, x0, y0, w, h)
    raise NotImplementedError


def _create_detector(architecture:str, device:int, activate_logits:bool) -> _AbstractDetector:
  # factory method
  init_funcs = {
    'yolov2': lambda: _AbstractYolov2(device, activate_logits),
    'ssd': lambda: _AbstractSSD(device, activate_logits),
    'yolov3': lambda: _AbstractYolov3(device, activate_logits)
  }
  return (init_funcs[architecture])()


class _AbstractYolov2(_AbstractDetector):
  def __init__(self, device:int, activate_logits:bool):
    super(_AbstractYolov2, self).__init__(device, activate_logits=activate_logits, input_w=608, input_h=608)
    self.num_cls = 80
    self.num_priors = 5
    self.num_non_cls_preds = 5 # (x off set, y off set, w, h, object logits)
    self.person_cls_id = 0
    self.cuda_device = torch.device("cuda:%i"%self.device)

  def _load_model(self, device:int) -> torch.nn.Module:
    cfg_file = FLAGS.yolov2_cfg_file
    weight_file = FLAGS.yolov2_weight_file
    yolov2 = Darknet(cfg_file)
    yolov2.load_weights(weight_file)
    return yolov2.eval().cuda(device)

  def _batched_preprocess(self, batched_images:torch.Tensor) -> torch.Tensor:
    return F.interpolate(batched_images, (self.input_h, self.input_w))

  def batched_forward_to_raw_output(self, batched_images:torch.Tensor):
    # [batch, num_priors*(num_non_cls_preds + num_cls), h, w]
    batched_logits = super(_AbstractYolov2, self).batched_forward_to_raw_output(batched_images)
    b = batched_logits.size(0)
    h = batched_logits.size(2)
    w = batched_logits.size(3)
    # [batch, num_priors, (num_non_cls_preds + num_cls), h, w]
    batched_logits = batched_logits.view(b, self.num_priors, self.num_non_cls_preds + self.num_cls,
                                         h, w)
    # [batch, h, w, num_priors, (num_non_cls_preds + num_cls)]
    batched_logits = batched_logits.permute(0, 3, 4, 1, 2).contiguous()
    # [batch, h*w*num_priors, (num_non_cls_preds + num_cls)]
    batched_logits = batched_logits.view(b, h*w*self.num_priors, self.num_non_cls_preds + self.num_cls)

    # [batch, h * w * num_priors]
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(b*self.num_priors, 1, 1).view(b, self.num_priors*h*w)\
      .cuda(self.cuda_device)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(b*self.num_priors, 1, 1).view(b, self.num_priors*h*w)\
      .cuda(self.cuda_device)
    # [batch, h * w * num_priors, 2]
    grid_xy = torch.stack([grid_x, grid_y], dim=2)
    cx_cy = torch.sigmoid(batched_logits[:, :, 0:2])
    cx_cy = cx_cy + grid_xy
    anchors_width_height = torch.tensor([w, h], dtype=torch.float).cuda(self.cuda_device)
    cx_cy = cx_cy / anchors_width_height
    return batched_logits, cx_cy

  def raw_output_to_objects_conf_cxcy(self, raw_output):
    logits, cx_cy = raw_output
    # [batch, h*w*num_priors]
    object_logits = logits[:, :, 4]
    objects = torch.sigmoid(object_logits) if self.activate_logits else object_logits
    return objects, cx_cy

  def raw_output_to_people_conf_cxcy(self, raw_output):
    logits, cx_cy = raw_output
    # [batch, h*w*num_priors, num_cls]
    class_logits = logits[:, :, self.num_non_cls_preds:self.num_non_cls_preds + self.num_cls]
    if self.activate_logits:
      class_confs = F.softmax(class_logits, dim=2)
      # [batch, h, w, num_priors]
      return class_confs[:, :, self.person_cls_id], cx_cy
    else:
      # [batch, h, w, num_priors]
      return class_logits[:, :, self.person_cls_id], cx_cy

  def single_image_nms_predictions(self, single_image:Image) -> np.ndarray:
    pass


class _AbstractSSD(_AbstractDetector):
  def __init__(self, device:int, activate_logits:bool):
    super(_AbstractSSD, self).__init__(device, activate_logits=activate_logits, input_w=300, input_h=300)
    cuda_device = torch.device('cuda:%i'%device)
    self.input_means = torch.tensor([123, 117, 104], dtype=torch.float, device=cuda_device).unsqueeze(-1).unsqueeze(-1)
    self.person_cls_id = 15

  def _load_model(self, device: int) -> torch.nn.Module:
    weightfile = FLAGS.ssd_weight_file
    voc_num_classes = 21
    # setting is_test to false since we dont need boxes, just confidence logits
    ssd = create_vgg_ssd(voc_num_classes, is_test=False)
    ssd.load(weightfile)
    return ssd.eval().cuda(device)

  def _batched_preprocess(self, batched_images: torch.Tensor) -> torch.Tensor:
    batched_images = batched_images * 255 - self.input_means
    return F.interpolate(batched_images, (self.input_h, self.input_w))

  def batched_forward_to_raw_output(self, batched_images:torch.Tensor):
    # confidences: [b, num_priors=4*sum(layersize), num_classes]
    # locations: [b, num_priors=4*sum(layersize), 4]
    confidences, locations = super(_AbstractSSD, self).batched_forward_to_raw_output(batched_images)
    # x_c, y_c, h, w
    locations = convert_locations_to_boxes(locations, self.model.priors, self.model.config.center_variance,
                                           self.model.config.size_variance)
    cx_cy = locations[:, :, 0:2]
    return confidences, cx_cy

  def raw_output_to_objects_conf_cxcy(self, raw_output):
    raise NotImplementedError

  def raw_output_to_people_conf_cxcy(self, raw_output):
    # confidences: [b, num_priors=4*sum(layersize), num_classes]
    # locations: [b, num_priors=4*sum(layersize), 4]
    confidences, cx_cy = raw_output
    if self.activate_logits:
      # [b, num_priors=4*sum(layersize), num_classes]
      confidences_activated = F.softmax(confidences, dim=2)
      return confidences_activated[:, :, self.person_cls_id], cx_cy
    else:
      return confidences[:, :, self.person_cls_id], cx_cy

  def single_image_nms_predictions(self, single_square_image: Image) -> np.ndarray:
    pass


class _AbstractYolov3(_AbstractDetector):
  def __init__(self, device:int, activate_logits:bool):
    super(_AbstractYolov3, self).__init__(device, activate_logits=activate_logits, input_w=608, input_h=608)
    self.num_cls = 80
    self.num_priors = 3
    self.num_non_cls_preds = 5 # (x off set, y off set, w, h, object logits)
    self.person_cls_id = 0
    self.w_h_tensor = torch.tensor([self.input_w, self.input_h], dtype=torch.float).cuda(torch.device("cuda:%i" % self.device))

  def _load_model(self, device: int) -> torch.nn.Module:
    cfg_file = FLAGS.yolov3_cfg_file
    weight_file = FLAGS.yolov3_weight_file
    yolov3 = Yolov3(cfg_file)
    yolov3.load_darknet_weights(weight_file)
    return yolov3.eval().cuda(device)

  def _inverse_sigmoid(self, activated_logits: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-7
    return torch.log((activated_logits + epsilon)/(-1 * activated_logits + 1 + epsilon))

  def _batched_preprocess(self, batched_images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(batched_images, (self.input_h, self.input_w))

  def batched_forward_to_raw_output(self, batched_images:torch.Tensor):
    raw = super(_AbstractYolov3, self).batched_forward_to_raw_output(batched_images)
    raw[:, :, 0:2] = raw[:, :, 0:2] / self.w_h_tensor # normalize locations
    return raw

  def raw_output_to_objects_conf_cxcy(self, raw_output: torch.Tensor):
    activated_objects = raw_output[:, :, 4]
    cx_cy = raw_output[:, :, 0:2]
    if self.activate_logits:
      return activated_objects, cx_cy
    else:
      return self._inverse_sigmoid(activated_objects), cx_cy

  def raw_output_to_people_conf_cxcy(self, raw_output: torch.Tensor):
    activated_persons = raw_output[:, :, self.num_non_cls_preds + self.person_cls_id]
    cx_cy = raw_output[:, :, 0:2]
    if self.activate_logits:
      return activated_persons, cx_cy
    else:
      return self._inverse_sigmoid(activated_persons), cx_cy

  def single_image_nms_predictions(self, single_square_image: Image) -> np.ndarray:
    pass


class EnsembleWeights(torch.nn.Module):
  def __init__(self, target_prior_weight:dict):
    super(EnsembleWeights, self).__init__()
    self.params = torch.nn.Parameter(data=torch.tensor([target_prior_weight[target] for target in target_prior_weight],
                                     dtype=torch.float))

  def forward(self) -> torch.Tensor:
    return F.softmax(self.params, dim=0)

