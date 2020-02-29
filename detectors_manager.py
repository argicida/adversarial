import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

# yolov2
from darknet import Darknet
# yolov3
from implementations.yolov3.models import Darknet as Yolov3
# ssd
from implementations.ssd.vision.ssd.vgg_ssd import create_vgg_ssd
from implementations.ssd.vision.utils.box_utils import convert_locations_to_boxes

from patch_utilities import PatchTransformApplier


# some of these have multiple output types, such as class confidence and object confidence, so multiple setting
SUPPORTED_TRAIN_DETECTORS = {'yolov2':3, # 1: class only, 2: object only, 3: object and class
                             'ssd':1, # 1: class
                             'yolov3':3 # 1: class only, 2: object only, 3: object and class
                            }


class Manager:
  def __init__(self, detector_device_mapping:dict, train_detector_settings:dict, test_detectors:list,
               activate_logits:bool=False):
    # all detectors in train_detector_settings and test_detectors must be mapped to a device in detector_device_mapping
    self.train_detector_settings = train_detector_settings
    self.test_detectors = test_detectors
    self.detector_instances = {}
    self.patch_appliers_by_device = {}
    for detector_name, cuda_device_id in detector_device_mapping.items():
      self.detector_instances[detector_name] = _create_detector(detector_name, device=cuda_device_id,
                                                                activate_logits=activate_logits)
      if cuda_device_id not in self.patch_appliers_by_device:
        self.patch_appliers_by_device[cuda_device_id] = PatchTransformApplier(cuda_device_id)

  def train_forward_propagate(self, input_images:torch.Tensor, patch_2d:torch.Tensor=None, labels_by_target:dict=None) \
        -> dict:
    # returns a dict of results, containing one or more logit maps for each target depending on setting
    # labels_by_target should contain tensors of labels for each target
    # pass in None for patch parameter if input_images are rendered from 3D model
    if patch_2d: assert labels_by_target is not None
    output_maps = {}
    copies_of_inputs_by_device = {}
    for target_name, target_setting in self.train_detector_settings.items():
      target = self.detector_instances[target_name]
      labels = labels_by_target[target_name]

      if target.device not in copies_of_inputs_by_device:
        device = torch.device('cuda:%i'%target.device)
        patch_copy = patch_2d.cuda(device=device, non_blocking=True) if patch_2d else None
        images_copy = input_images.cuda(device=device, non_blocking=True)
        copies_of_inputs_by_device[target.device] = (images_copy, patch_copy)
      else:
        images_copy, patch_copy = copies_of_inputs_by_device[target.device]
      detector_input = self.patch_appliers_by_device[target.device](patch_copy, images_copy, labels) \
          if patch_2d else images_copy

      logits = target.batched_forward_to_raw_output(detector_input)
      if target_setting == 1:
        output_maps[target_name] = [target.raw_output_to_persons(logits)]
      elif target_setting == 2:
        output_maps[target_name] = [target.raw_output_to_objects(logits)]
      else:
        output_maps[target_name] = [target.raw_output_to_persons(logits), target.raw_output_to_objects(logits)]
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
  def __init__(self, device:int, activate_logits:bool):
    self.device = device
    self.model = self._load_model(device)
    self.activate_logits = activate_logits
    return

  def batched_forward_to_raw_output(self, batched_images:torch.Tensor) -> torch.Tensor:
    return self.model(self._batched_preprocess(batched_images))

  def _load_model(self, device:int) -> torch.nn.Module:
    # returns an nn.module that forward pass batched images into a tensor of logits
    # which can be turned into confidence scores
    raise NotImplementedError

  def _batched_preprocess(self, batched_images: torch.Tensor):
    # preprocess images for the particular model
    raise NotImplementedError

  def raw_output_to_objects(self, raw_output):
    raise NotImplementedError

  def raw_output_to_persons(self, raw_output):
    raise NotImplementedError

  def single_image_nms_predictions(self, single_square_image:Image) -> np.ndarray:
    # used for testing, which goes through images one by one
    # returns array of normalized (is_person, x0, y0, w, h)
    raise NotImplementedError


def _create_detector(architecture:str, device:int, activate_logits:bool) -> _AbstractDetector:
  # factory method
  init_funcs = {
    'yolov2': _AbstractYolov2.__init__,
    'ssd': _AbstractSSD.__init__,
    'yolov3': _AbstractYolov3.__init__
  }
  return (init_funcs[architecture])(device, activate_logits)


class _AbstractYolov2(_AbstractDetector):
  def __init__(self, device:int, activate_logits:bool):
    super(_AbstractYolov2, self).__init__(device, activate_logits=activate_logits)
    self.num_cls = 80
    self.num_priors = 5
    self.num_non_cls_preds = 5 # (x off set, y off set, w, h, object logits)
    self.person_cls_id = 0

  def _load_model(self, device:int) -> torch.nn.Module:
    cfg_file = "cfg/yolov2.cfg"
    weight_file = "weights/yolov2.weights"
    yolov2 = Darknet(cfg_file)
    yolov2.load_weights(weight_file)
    return yolov2.eval().cuda(device)

  def _batched_preprocess(self, batched_images:torch.Tensor) -> torch.Tensor:
    return F.interpolate(batched_images, (608, 608))

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

    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(b*self.num_priors, 1, 1).view(b, self.num_priors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(b*self.num_priors, 1, 1).view(b, self.num_priors*h*w).cuda()
    grid_xy = torch.stack([grid_x, grid_y], dim=2)

    cx_cy = torch.sigmoid(batched_logits[:, 0:2])
    cx_cy = cx_cy + grid_xy
    return batched_logits, cx_cy

  def raw_output_to_objects(self, raw_output):
    logits, cx_cy = raw_output
    # [batch, h*w*num_priors]
    object_logits = logits[:, :, 4]
    objects = torch.sigmoid(object_logits) if self.activate_logits else object_logits
    return objects, cx_cy

  def raw_output_to_persons(self, raw_output):
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
    super(_AbstractSSD, self).__init__(device, activate_logits=activate_logits)
    cuda_device = torch.device('cuda:%i'%device)
    self.input_means = torch.tensor([123, 117, 104], dtype=torch.float, device=cuda_device).unsqueeze(-1).unsqueeze(-1)
    self.person_cls_id = 15

  def _load_model(self, device: int) -> torch.nn.Module:
    weightfile = "./implementations/ssd/models/vgg16-ssd-mp-0_7726.pth"
    voc_num_classes = 21
    # setting is_test to false since we dont need boxes, just confidence logits
    ssd = create_vgg_ssd(voc_num_classes, is_test=False)
    ssd.load(weightfile)
    return ssd.eval().cuda(device)

  def _batched_preprocess(self, batched_images: torch.Tensor) -> torch.Tensor:
    batched_images = batched_images * 255 - self.input_means
    return F.interpolate(batched_images, (300, 300))

  def batched_forward_to_raw_output(self, batched_images:torch.Tensor):
    # confidences: [b, num_priors=4*sum(layersize), num_classes]
    # locations: [b, num_priors=4*sum(layersize), 4]
    confidences, locations = super(_AbstractSSD, self).batched_forward_to_raw_output(batched_images)
    # x_c, y_c, h, w
    locations = convert_locations_to_boxes(locations, self.model.priors, self.model.config.center_variance,
                                           self.model.config.size_variance)
    cx_cy = locations[:, :, 0:2]
    return confidences, cx_cy

  def raw_output_to_objects(self, raw_output):
    raise NotImplementedError

  def raw_output_to_persons(self, raw_output):
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
    super(_AbstractYolov3, self).__init__(device, activate_logits=activate_logits)
    self.num_cls = 80
    self.num_priors = 3
    self.num_non_cls_preds = 5 # (x off set, y off set, w, h, object logits)
    self.person_cls_id = 0

  def _load_model(self, device: int) -> torch.nn.Module:
    cfg_file = "./implementations/yolov3/config/yolov3.cfg"
    weight_file = "./implementations/yolov3/weights/yolov3.weights"
    yolov3 = Yolov3(cfg_file)
    yolov3.load_darknet_weights(weight_file)
    return yolov3.eval().cuda(device)

  def _inverse_sigmoid(self, activated_logits: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-7
    return torch.log((activated_logits + epsilon)/(-1 * activated_logits + 1 + epsilon))

  def _batched_preprocess(self, batched_images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(batched_images, (608, 608))

  def raw_output_to_objects(self, raw_output: torch.Tensor):
    activated_objects = raw_output[:, :, 4]
    cx_cy = raw_output[:, :, 0:2]
    if self.activate_logits:
      return activated_objects, cx_cy
    else:
      return self._inverse_sigmoid(activated_objects), cx_cy

  def raw_output_to_persons(self, raw_output: torch.Tensor):
    activated_persons = raw_output[:, :, self.num_non_cls_preds + self.person_cls_id]
    cx_cy = raw_output[:, :, 0:2]
    if self.activate_logits:
      return activated_persons, cx_cy
    else:
      return self._inverse_sigmoid(activated_persons), cx_cy

  def single_image_nms_predictions(self, single_square_image: Image) -> np.ndarray:
    pass
