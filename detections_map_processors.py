"""
different functions for processing 2d maps of class/object confidences/logits into scalars for the loss function
"""
import torch
import sys


def uninformed_detections_processor_choices() -> list:
  return list(_uninformed_functions.keys())


def informed_detections_processor_choices() -> list:
  return list(_informed_functions.keys())


def processor_choices() -> dict:
  return {'uninformed (no gt needed)':uninformed_detections_processor_choices(),
          'informed (requires gt)':informed_detections_processor_choices()}


def check_detector_output_processor_exists(choice:str):
  if choice not in _informed_functions.keys() and choice not in _uninformed_functions.keys():
    print("choice of detector output processor does not exist, your options:\n" + str(processor_choices()),
          file=sys.stderr)
    sys.exit(1)


def process(output_map:torch.Tensor, choice:str, gt_boxes:torch.Tensor=None) -> torch.Tensor:
  """
  :param output_map: w*h map of floats corresponding to predictions from different regions in input image
  :param choice: choice of processor
  :param gt_boxes: gt boxes consisted of [is_not_person, x_center, y_center, width, height], with [1:] rescaled
                  required for informed processors. keep it on cpu for faster processing here
  :return: scalar tensor
  """
  check_detector_output_processor_exists(choice)
  if choice in _informed_functions.keys():
    if gt_boxes is None:
      print("ground truth bounding boxes required for informed output processing", file=sys.stderr)
      sys.exit(1)
    fn = _informed_functions[choice]
    return fn(output_map, gt_boxes)
  if choice in _uninformed_functions.keys():
    fn = _uninformed_functions[choice]
    return fn(output_map)


def _map_avg(output_map:torch.Tensor) -> torch.Tensor:
  """
  avg of all the values
  :param output_map:
  :return:
  """
  return torch.mean(output_map, dim=1)


def _map_max(output_map:torch.Tensor) -> torch.Tensor:
  """
  max of all the values
  :param output_map:
  :return:
  """
  return torch.max(output_map, dim=1)[0]


def _detection_max(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  max of all the values that are in person bounding box ground truths
  :param output_map:
  :param gt_boxes:
  :return:
  """
  masked_output = _masked_output_maps_with_person_boxes(output_map, gt_boxes)
  return torch.max(masked_output, dim=1)[0]


def _detection_avg(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  avg of all the values that are in person bounding box ground truths
  :param output_map:
  :param gt_boxes:
  :return:
  """
  masked_output = _masked_output_maps_with_person_boxes(output_map, gt_boxes)
  return torch.mean(masked_output, dim=1)


def _detection_max_avg(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  avg of maxes of the values that are in each person bounding box ground truth respectively
  :param output_map:
  :param gt_boxes:
  :return:
  """
  map_shape = output_map.shape[1:]
  means = torch.empty(map_shape[0], dtype=torch.float)
  for i, image_output in enumerate(output_map):
    boxes = gt_boxes[i]
    human_boxes = boxes[boxes[:, 0] == 0]
    masks = [_produce_bitwise_mask_for_box(box, map_shape) for box in human_boxes]
    means[i] = torch.mean(torch.stack([torch.max(image_output[person_mask])[0] for person_mask in masks]))
  return means


def _produce_bitwise_mask_for_box(box, map_shape) -> torch.Tensor:
  W = map_shape[0]
  H = map_shape[1]
  x = box[1] * W
  y = box[2] * H
  w = box[3] * W / 2
  h = box[4] * H / 2
  mask = torch.zeros(map_shape, dtype=torch.bool)
  mask[int(x - w):int(x + w) + 1, int(y - h):int(y + h) + 1] = True
  return mask


def _bitwise_union_boxes_not_batched(boxes:torch.Tensor, map_shape) -> torch.Tensor:
  human_boxes = boxes[boxes[:, 0] == 0]
  return torch.sum(torch.stack([_produce_bitwise_mask_for_box(box, map_shape) for box in human_boxes]),
                    dim=0, dtype=torch.bool)


def _masked_output_maps_with_person_boxes(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  map_shape = output_map.shape[1:]
  masks = torch.stack([_bitwise_union_boxes_not_batched(boxes, map_shape) for boxes in gt_boxes]).cuda().float()
  return torch.mul(output_map, masks)


# functions that don't make use of ground truth, and only uses detector output
# batch_of_output_map -> batch_of_scalar
_uninformed_functions = {"map_avg":_map_avg,
                         "map_max":_map_max}

# functions that leverage knowledge of ground truth
# batch_of_output_map, batch_of_batch_of_gt_boxes -> batch_of_scalar
_informed_functions = {"det_max":_detection_max,
                       "det_avg":_detection_avg,
                       "det_max_avg":_detection_max_avg}
