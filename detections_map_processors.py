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


def process(confidences:torch.Tensor, locations:torch.Tensor, choice:str, gt_boxes:torch.Tensor=None) -> torch.Tensor:
  """
  :param confidences: [batch size, num_predictions] batch of floats corresponding to predictions
  :param locations: [batch size, num_predictions, 2] same shape as confidences with one more axis for center_x, center_y
  :param choice: choice of processor
  :param gt_boxes: gt boxes for person bounding boxes consisted of [is_person, x_0, y_0, x_1, y_1] (relative to size of pics)
  :return: [batch size]
  """
  check_detector_output_processor_exists(choice)
  if choice in _informed_functions.keys():
    if gt_boxes is None:
      print("ground truth bounding boxes required for informed output processing", file=sys.stderr)
      sys.exit(1)
    fn = _informed_functions[choice]
    return fn(confidences, locations, gt_boxes)
  if choice in _uninformed_functions.keys():
    fn = _uninformed_functions[choice]
    return fn(confidences)


def _avg(confidences:torch.Tensor) -> torch.Tensor:
  """
  avg of all the values
  :param confidences: [batch size, num_predictions] batch of floats corresponding to predictions
  :return: [batch size]
  """
  return torch.mean(confidences, dim=1)


def _max(confidences:torch.Tensor) -> torch.Tensor:
  """
  max of all the values
  :param confidences: [batch size, num_predictions] batch of floats corresponding to predictions
  :return: [batch size]
  """
  return torch.max(confidences, dim=1)[0]


# functions that don't make use of ground truth, and only uses detector output
# batch_of_output_map -> batch_of_scalar
_uninformed_functions = {"map_avg":_avg,
                         "map_max":_max}


def _compare_locations_to_gt_boxes(locations:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  returned boolean tensor is always detached from autograd graph, which is perfect for our purpose
  :param locations: [batch size, num_predictions, 2] same shape as confidences with one more axis for center_x, center_y
  :param gt_boxes: [batch size, num_labels, 5] is_person, x0, y0, x1, y1
  :return: [batch size, num_predictions, num_labels] whether a prediction lies within a person label
  """
  batch_size = locations.shape[0]
  num_predictions = locations.shape[1]
  num_labels = gt_boxes.shape[1]
  # [batch size, num_predictions, num_labels, 2]
  locations_broadcast = locations.view(batch_size, num_predictions, 1, 2).expand(-1, -1, num_labels, -1)
  # [batch size, num_predictions, num_labels, 5]
  boxes_broadcast = gt_boxes.view(batch_size, 1, num_labels, 5).expand(-1, num_predictions, -1, -1)
  # [batch size, num_predictions, num_labels]
  compared_to_person_label = (boxes_broadcast[:, :, :, 0] == 0)
  x0 = boxes_broadcast[:, :, :, 1]
  x1 = boxes_broadcast[:, :, :, 3]
  y0 = boxes_broadcast[:, :, :, 2]
  y1 = boxes_broadcast[:, :, :, 4]
  cx = locations_broadcast[:, :, :, 0]
  cy = locations_broadcast[:, :, :, 1]
  x_within_box = ((cx >= x0) & (cx <= x1))
  y_within_box = ((cy >= y0) & (cy <= y1))
  return compared_to_person_label & x_within_box & y_within_box


def _detection_max(confidences:torch.Tensor, locations:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  :param confidences: [batch size, num_predictions] batch of floats corresponding to predictions
  :param locations: [batch size, num_predictions, 2] same shape as confidences with one more axis for center_x, center_y
  :param gt_boxes: [batch size, num_labels, 5] is_person, x0, y0, x1, y1
  """
  # [batch size, num_predictions, num_labels]
  locations_compared_to_person_labels = _compare_locations_to_gt_boxes(locations, gt_boxes)
  # [batch size, num_predictions] dtype=torch.bool detached
  confidence_within_a_person_label = torch.sum(locations_compared_to_person_labels, dim=2, dtype=torch.bool)
  # [batch size, num_predictions] dtype=torch.float
  confidences_masked = confidences * confidence_within_a_person_label.float()
  # [batch size]
  return torch.max(confidences_masked, dim=1)[0]


def _detection_avg(confidences:torch.Tensor, locations:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  :param confidences: [batch size, num_predictions] batch of floats corresponding to predictions
  :param locations: [batch size, num_predictions, 2] same shape as confidences with one more axis for center_x, center_y
  :param gt_boxes: [batch size, num_labels, 4] x0, y0, x1, y1
  """
  # [batch size, num_predictions, num_labels]
  locations_compared_to_person_labels = _compare_locations_to_gt_boxes(locations, gt_boxes)
  # [batch size, num_predictions] dtype=torch.bool detached
  confidence_within_a_person_label = torch.sum(locations_compared_to_person_labels, dim=2, dtype=torch.bool)
  # [batch size, num_predictions] dtype=torch.float
  confidences_masked = confidences * confidence_within_a_person_label.float()
  # [batch size]
  return torch.mean(confidences_masked, dim=1)[0]


def _detection_max_avg(confidences:torch.Tensor, locations:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
  """
  :param confidences: [batch size, num_predictions] batch of floats corresponding to predictions
  :param locations: [batch size, num_predictions, 2] same shape as confidences with one more axis for center_x, center_y
  :param gt_boxes: [batch size, num_labels, 4] x0, y0, x1, y1
  """
  batch_size = locations.shape[0]
  num_predictions = locations.shape[1]
  num_labels = gt_boxes.shape[1]
  # [batch size, num_predictions, num_labels]
  locations_compared_to_person_labels = _compare_locations_to_gt_boxes(locations, gt_boxes)
  confidences_broadcast = confidences.view(batch_size, num_predictions, 1)
  confidences_matched_with_person = confidences_broadcast * locations_compared_to_person_labels
  # [batch size, num_labels]
  person_max_confidence = torch.max(confidences_matched_with_person, dim=1)[0]
  # [batch size]
  return torch.mean(person_max_confidence, dim=1)


# functions that leverage knowledge of ground truth
# batch_of_output_map, batch_of_batch_of_gt_boxes -> batch_of_scalar
_informed_functions = {"det_max":_detection_max,
                       "det_avg":_detection_avg,
                       "det_max_avg":_detection_max_avg}


#### grid based functions
# def _detection_max(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
#   """
#   max of all the values that are in person bounding box ground truths
#   :param output_map:
#   :param gt_boxes:
#   :return:
#   """
#   masked_output = _masked_output_maps_with_person_boxes(output_map, gt_boxes)
#   b = output_map.shape[0]
#   return torch.max(masked_output.view(b, masked_output.numel()//b), dim=1)[0]
#
#
# def _detection_avg(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
#   """
#   avg of all the values that are in person bounding box ground truths
#   :param output_map:
#   :param gt_boxes:
#   :return:
#   """
#   reduced_axes = list(range(1, output_map.ndimension()))
#   masked_output = _masked_output_maps_with_person_boxes(output_map, gt_boxes)
#   return torch.mean(masked_output, dim=reduced_axes)
#
#
# def _detection_max_avg(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
#   """
#   avg of maxes of the values that are in each person bounding box ground truth respectively
#   :param output_map:
#   :param gt_boxes:
#   :return:
#   """
#   batch_size = output_map.shape[0]
#   map_shape = output_map.shape[1:3] # [h, w]
#   means = torch.empty(batch_size, dtype=torch.float)
#   for i, image_output in enumerate(output_map):
#     boxes = gt_boxes[i]
#     human_boxes = boxes[boxes[:, 0] == 0]
#     masks = [_produce_bitwise_mask_for_box(box, map_shape) for box in human_boxes]
#     means[i] = torch.mean(torch.stack([torch.max(image_output[person_mask])[0] for person_mask in masks]))
#   return means
#
#
# def _produce_bitwise_mask_for_box(box, map_shape) -> torch.Tensor:
#   W = map_shape[1]
#   H = map_shape[0]
#   x = box[1] * W
#   y = box[2] * H
#   w = box[3] * W / 2
#   h = box[4] * H / 2
#   mask = torch.zeros(map_shape, dtype=torch.bool)
#   mask[max(int(y - h), 0):min(int(y + h) + 1, H), max(int(x - w), 0):min(int(x + w) + 1, W)] = True
#   return mask
#
#
# def _bitwise_union_boxes_not_batched(boxes:torch.Tensor, map_shape) -> torch.Tensor:
#   human_boxes = boxes[boxes[:, 0] == 0]
#   return torch.sum(torch.stack([_produce_bitwise_mask_for_box(box, map_shape) for box in human_boxes]),
#                     dim=0, dtype=torch.bool)
#
#
# def _masked_output_maps_with_person_boxes(output_map:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
#   map_shape = output_map.shape[1:3]
#   masks = torch.stack([_bitwise_union_boxes_not_batched(boxes, map_shape) for boxes in gt_boxes]).cuda().float()
#   num_extra_axes = output_map.ndimension()-3
#   expanded_masks = masks[(...,) + (None,) * num_extra_axes]
#   return torch.mul(output_map, expanded_masks)
