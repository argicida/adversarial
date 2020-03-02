import torch
import os
from absl import app
from torch.utils.data import DataLoader
from torchvision import transforms

from cli_config import FLAGS
from inria import InriaDataset
from detectors_manager import SUPPORTED_TRAIN_DETECTORS, Manager
from patch_utilities import SquarePatch, NPSCalculator, TotalVariationCalculator
from detections_map_processors import check_detector_output_processor_exists, process


def main(argv):
  # INITIALIZATION
  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)
  check_detector_output_processor_exists(FLAGS.confidence_processor)
  cuda_device_id = 0 # we only use a single GPU at the moment
  flags_dict = FLAGS.flag_values_dict()
  target_settings = {}
  target_devices = {}
  for candidate in SUPPORTED_TRAIN_DETECTORS:
    key = "train_%s"%candidate
    if key in flags_dict:
      setting = flags_dict[key]
      if setting != 0:
        target_settings[candidate] = flags_dict[key]
        target_devices[candidate] = cuda_device_id
  train_loader = DataLoader(InriaDataset(FLAGS.inria_dir, FLAGS.max_labs, FLAGS.init_size, list(target_settings.keys())),
                            batch_size=FLAGS.bs, num_workers=FLAGS.num_workers, shuffle=True)
  iterations_per_epoch = len(train_loader)
  targets_manager = Manager(target_devices, target_settings, activate_logits=FLAGS.activate_logits,
                            debug_autograd=FLAGS.debug_autograd, debug_device=FLAGS.debug_device)
  patch_module_gpu = SquarePatch(patch_size=FLAGS.patch_square_length, typ=FLAGS.start_patch).cuda(cuda_device_id)
  nps_gpu = NPSCalculator(FLAGS.printable_vals_filepath, FLAGS.patch_square_length).cuda(cuda_device_id)
  tv_gpu = TotalVariationCalculator().cuda(cuda_device_id)
  optimizer = torch.optim.Adam(patch_module_gpu.parameters(), lr=FLAGS.lr)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=FLAGS.plateau_patience)

  num_targets = len(target_settings)
  static_ensemble_weights_gpu = torch.full([num_targets], fill_value=1/num_targets,
                                       device=torch.device('cuda:%i'%cuda_device_id))

  # TRAINING
  for epoch in range(FLAGS.n_epochs):
    epoch_total_loss_sum = 0.0
    epoch_unweighted_detector_loss_sum = {detector_name:0.0 for detector_name in target_settings}
    epoch_weighted_detection_loss_sum = 0.0
    epoch_weighted_nonprintability_loss_sum = 0.0
    epoch_weighted_patch_variation_loss_sum = 0.0
    for batch_index, (images, labels_dict) in enumerate(train_loader):
      with torch.autograd.set_detect_anomaly(FLAGS.debug_autograd):
        adv_patch_gpu = patch_module_gpu()
        outputs_by_target = targets_manager.train_forward_propagate(images, labels_by_target=labels_dict,
                                                                    patch_2d=adv_patch_gpu)
        target_confidences = []
        for target_name in outputs_by_target:
          setting = target_settings[target_name]
          labels = labels_dict[target_name]
          if setting is 1 or setting is 2:
            # [batch_size, num_predictions] confidence
            # [batch_size, num_predictions, 2] cx_cy
            confidences, cx_cy, labels_on_device = outputs_by_target[target_name][0]
            # [1]
            extracted_confidence = torch.mean(process(confidences, cx_cy, FLAGS.confidence_processor, labels_on_device))
          elif setting is 3:
            object_coef = flags_dict['%s_object_weight'%target_name]
            person_coef = 1 - object_coef
            # [batch_size, num_predictions] confidence
            # [batch_size, num_predictions, 2] cx_cy
            person_confidences, person_cx_cy, labels_on_device = outputs_by_target[target_name][0]
            object_confidences, object_cx_cy, labels_on_device = outputs_by_target[target_name][1]
            # [batch_size]
            extracted_person_confidences = process(person_confidences, person_cx_cy, FLAGS.confidence_processor,
                                                   labels_on_device)
            extracted_object_confidences = process(object_confidences, object_cx_cy, FLAGS.confidence_processor,
                                                   labels_on_device)
            # [1]
            extracted_confidence = object_coef * torch.mean(extracted_object_confidences)\
                                   + person_coef * torch.mean(extracted_person_confidences)
          else:
            assert False
          target_confidences.append(extracted_confidence)
          print(target_name, extracted_confidence)
          epoch_unweighted_detector_loss_sum[target_name] += extracted_confidence.detach().cpu().numpy()
        # [num_target]
        target_confidences = torch.stack(target_confidences)
        detection_loss_gpu = torch.matmul(static_ensemble_weights_gpu, target_confidences)
        printability_loss_gpu = FLAGS.lambda_nps * nps_gpu(adv_patch_gpu)
        patch_variation_loss_gpu = FLAGS.lambda_tv * tv_gpu(adv_patch_gpu)
        total_loss_gpu = detection_loss_gpu + printability_loss_gpu + patch_variation_loss_gpu

        # GRADIENT UPDATE
        total_loss_gpu.backward()
        optimizer.step()
        optimizer.zero_grad()

        # BATCH LOGGING
        epoch_total_loss_sum += total_loss_gpu.detach().cpu().numpy()
        epoch_weighted_detection_loss_sum += detection_loss_gpu.detach().cpu().numpy()
        epoch_weighted_nonprintability_loss_sum += printability_loss_gpu.detach().cpu().numpy()
        epoch_weighted_patch_variation_loss_sum += patch_variation_loss_gpu.detach().cpu().numpy()

    # EPOCH LOGGING & LR SCHEDULING
    epoch_total_loss_mean = epoch_total_loss_sum/iterations_per_epoch
    lr_scheduler.step(epoch_total_loss_mean)
    epoch_unweighted_detector_loss_mean = {detector_name:epoch_unweighted_detector_loss_sum[detector_name]
                                                         /iterations_per_epoch
                                           for detector_name in target_settings}
    epoch_weighted_detection_loss_mean = epoch_weighted_detection_loss_sum/iterations_per_epoch
    epoch_weighted_nonprintability_loss_mean = epoch_weighted_nonprintability_loss_sum/iterations_per_epoch
    epoch_weighted_patch_variation_loss_mean = epoch_weighted_patch_variation_loss_sum/iterations_per_epoch
    print('  EPOCH NR: ', epoch),
    print('EPOCH LOSS: %.3f'%epoch_total_loss_mean)
    print('  DET LOSS: %.3f'%epoch_weighted_detection_loss_mean)
    print('  NPS LOSS: %.3f'%epoch_weighted_nonprintability_loss_mean)
    print('   TV LOSS: %.3f'%epoch_weighted_patch_variation_loss_mean)
    for detector_name in epoch_unweighted_detector_loss_mean:
      print('%s: %.3f'%(detector_name, epoch_unweighted_detector_loss_mean[detector_name]))
    print()
    im = transforms.ToPILImage('RGB')(adv_patch_gpu.cpu())
    im.save(os.path.join(FLAGS.logdir, "patch.png"), "PNG")


if __name__ == '__main__':
  app.run(main)
