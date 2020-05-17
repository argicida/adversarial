import torch
import os
import json
import numpy as np
from absl import app
from torch.utils.data import DataLoader
from torchvision import transforms
from ray.tune import track
from tensorboardX import SummaryWriter

from cli_config import FLAGS
from inria import InriaDataset
from detectors_manager import SUPPORTED_TRAIN_DETECTORS, Manager, EnsembleWeights
from patch_utilities import SquarePatch, NPSCalculator, TotalVariationCalculator
from detections_map_processors import check_detector_output_processor_exists, process
from test_patch import test_on_all_detectors, SUPPORTED_TEST_DETECTORS


def train():
  # INITIALIZATION
  if FLAGS.verbose: print("Initializing")
  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)
  flags_dict = FLAGS.flag_values_dict()
  with open(os.path.join(FLAGS.logdir, 'flags.json'), 'w') as fp:
    json.dump(flags_dict, fp)
  tensorboard_writer = SummaryWriter(logdir=FLAGS.logdir) if FLAGS.tensorboard_batch or FLAGS.tensorboard_epoch else None
  check_detector_output_processor_exists(FLAGS.confidence_processor)
  cuda_device_id = 0 # we only use a single GPU at the moment
  target_settings = {}
  target_devices = {}
  target_prior_weight = {}
  for candidate in SUPPORTED_TRAIN_DETECTORS:
    setting_key = "train_%s"%candidate
    prior_weight_key = "%s_prior_weight"%candidate
    setting = flags_dict[setting_key]
    if setting != 0:
      target_settings[candidate] = flags_dict[setting_key]
      target_prior_weight[candidate] = flags_dict[prior_weight_key]
      target_devices[candidate] = cuda_device_id
  train_loader = DataLoader(InriaDataset(FLAGS.inria_train_dir, FLAGS.max_labs, FLAGS.init_size,
                                         list(target_settings.keys())),
                            batch_size=FLAGS.mini_bs, num_workers=FLAGS.num_workers, shuffle=True)
  iterations_per_epoch = len(train_loader)
  targets_manager = Manager(target_devices, target_settings, activate_logits=FLAGS.activate_logits,
                            debug_autograd=FLAGS.debug_autograd, debug_device=FLAGS.debug_device,
                            debug_coords=FLAGS.debug_coords, plot_patches=FLAGS.plot_patches)
  nps_gpu = NPSCalculator(FLAGS.printable_vals_filepath, FLAGS.patch_square_length).cuda(cuda_device_id)
  tv_gpu = TotalVariationCalculator().cuda(cuda_device_id)

  patch_module_gpu, patch_optimizer, patch_lr_scheduler, ensemble_weights_module_gpu, ensemble_weights_optimizer \
      = allocate_memory_for_stateful_components(cuda_device_id, target_prior_weight)
  norm_prior_ensemble_weights_gpu = ensemble_weights_module_gpu().detach().clone()
  start_epoch = continue_from_checkpoint_if_exists_and_get_epoch(patch_module_gpu, patch_optimizer, patch_lr_scheduler,
                                                                ensemble_weights_module_gpu, ensemble_weights_optimizer)
  if FLAGS.verbose: print("Session Initialized")
  
  batch_size = FLAGS.mini_bs * FLAGS.num_mini
  n_batches = int(len(train_loader) / batch_size)
  n_mini_batches = FLAGS.num_mini

  # TRAINING
  target_extracted_confidences_gpu_dict = {}
  for epoch in range(start_epoch, FLAGS.n_epochs):
    if FLAGS.verbose: print('  EPOCH NR: ', epoch)
    epoch_total_loss_sum = 0.0
    epoch_unweighted_detector_loss_sum = {detector_name:0.0 for detector_name in target_settings}
    epoch_ensemble_weights_sum = {detector_name:0.0 for detector_name in target_settings}
    epoch_weighted_detection_loss_sum = 0.0
    epoch_weighted_printability_loss_sum = 0.0
    epoch_weighted_patch_variation_loss_sum = 0.0
    minibatch_iterator = iter(train_loader)
    for nth_batch in range(n_batches):
      batch_extracted_confidence = {detector_name:0.0 for detector_name in target_settings}
      batch_ensemble_weights = {detector_name:0.0 for detector_name in target_settings}
      batch_total_loss_sum = 0.0
      batch_detection_loss_sum = 0.0
      batch_printability_loss_sum = 0.0
      batch_patch_variation_loss_sum = 0.0
      mini_batches = []
      with torch.autograd.set_detect_anomaly(FLAGS.debug_autograd):
        if FLAGS.minimax:
          normalized_ensemble_weights_gpu = ensemble_weights_module_gpu()
          if FLAGS.verbose: print("Before Max Step, ", normalized_ensemble_weights_gpu)
          for nth_mini_batch in range(FLAGS.mini_bs):
            images, normed_labels_dict = next(minibatch_iterator)
            mini_batches.append([images,normed_labels_dict])
            adv_patch_gpu = patch_module_gpu()
            outputs_by_target = targets_manager.train_forward_propagate(images, labels_by_target=normed_labels_dict,
                                                                        patch_2d=adv_patch_gpu)
            for target_name in outputs_by_target:
              setting = target_settings[target_name]
              if setting is 1 or setting is 2:
                # [batch_size, num_predictions] confidence
                # [batch_size, num_predictions, 2] cx_cy
                confidences, cx_cy, normed_labels_on_device = outputs_by_target[target_name][0]
                # [1]
                extracted_confidence = torch.mean(process(confidences, cx_cy, FLAGS.confidence_processor,
                                                          normed_labels_on_device))
              elif setting is 3:
                object_coef = flags_dict['%s_object_weight'%target_name]
                person_coef = 1 - object_coef
                # [batch_size, num_predictions] confidence
                # [batch_size, num_predictions, 2] cx_cy
                person_confidences, person_cx_cy, normed_labels_on_device = outputs_by_target[target_name][0]
                object_confidences, object_cx_cy, normed_labels_on_device = outputs_by_target[target_name][1]
                # [batch_size]
                extracted_person_confidences = process(person_confidences, person_cx_cy, FLAGS.confidence_processor,
                                                       normed_labels_on_device)
                extracted_object_confidences = process(object_confidences, object_cx_cy, FLAGS.confidence_processor,
                                                       normed_labels_on_device)
                # [1]
                extracted_confidence = object_coef * torch.mean(extracted_object_confidences)\
                                       + person_coef * torch.mean(extracted_person_confidences)
              else:
                assert False
              target_extracted_confidences_gpu_dict[target_name] = extracted_confidence
            # [num_target]
            target_extracted_confidences_tensor = torch.stack(list(target_extracted_confidences_gpu_dict.values()))
            if FLAGS.verbose: print("Dict Out", target_extracted_confidences_gpu_dict,
                                    "Stacked Out, ", target_extracted_confidences_tensor)

            # MAX STEP
            normalized_ensemble_weights_gpu = ensemble_weights_module_gpu()
            # prevent backpropagation through detectors during max step to save computation - only needed during min step
            detached_target_extracted_confidences_tensor = target_extracted_confidences_tensor.detach()
            # [1]
            detection_loss_gpu = torch.matmul(normalized_ensemble_weights_gpu,
                                              detached_target_extracted_confidences_tensor)
            prior_l2_distance = torch.dist(normalized_ensemble_weights_gpu, norm_prior_ensemble_weights_gpu, p=2)
            max_loss = (-detection_loss_gpu + FLAGS.minimax_gamma * prior_l2_distance) / n_mini_batches
            max_loss.backward()
          # MAX WEIGHT UPDATE
          ensemble_weights_optimizer.step()
          ensemble_weights_optimizer.zero_grad()
          normalized_ensemble_weights_gpu = ensemble_weights_module_gpu()
          if FLAGS.verbose: print("After Max Step, ", normalized_ensemble_weights_gpu)
        # MIN STEP
        for nth_mini_batch in range(FLAGS.mini_bs):
          if FLAGS.minimax:
            images, normed_labels_dict = mini_batches[nth_mini_batch]
          else:
            images, normed_labels_dict = next(minibatch_iterator)
          adv_patch_gpu = patch_module_gpu()
          outputs_by_target = targets_manager.train_forward_propagate(images, labels_by_target=normed_labels_dict,
                                                                      patch_2d=adv_patch_gpu)
          for target_name in outputs_by_target:
            setting = target_settings[target_name]
            if setting is 1 or setting is 2:
              # [batch_size, num_predictions] confidence
              # [batch_size, num_predictions, 2] cx_cy
              confidences, cx_cy, normed_labels_on_device = outputs_by_target[target_name][0]
              # [1]
              extracted_confidence = torch.mean(process(confidences, cx_cy, FLAGS.confidence_processor,
                                                        normed_labels_on_device))
            elif setting is 3:
              object_coef = flags_dict['%s_object_weight'%target_name]
              person_coef = 1 - object_coef
              # [batch_size, num_predictions] confidence
              # [batch_size, num_predictions, 2] cx_cy
              person_confidences, person_cx_cy, normed_labels_on_device = outputs_by_target[target_name][0]
              object_confidences, object_cx_cy, normed_labels_on_device = outputs_by_target[target_name][1]
              # [batch_size]
              extracted_person_confidences = process(person_confidences, person_cx_cy, FLAGS.confidence_processor,
                                                     normed_labels_on_device)
              extracted_object_confidences = process(object_confidences, object_cx_cy, FLAGS.confidence_processor,
                                                     normed_labels_on_device)
              # [1]
              extracted_confidence = object_coef * torch.mean(extracted_object_confidences)\
                                     + person_coef * torch.mean(extracted_person_confidences)
            else:
              assert False
            target_extracted_confidences_gpu_dict[target_name] = extracted_confidence
            # ACCUMULATE BATCH STATISTIC
            batch_extracted_confidence[target_name] += extracted_confidence / n_mini_batches
          # [num_target]
          target_extracted_confidences_tensor = torch.stack(list(target_extracted_confidences_gpu_dict.values()))
          if FLAGS.verbose: print("Dict Out", target_extracted_confidences_gpu_dict,
                                  "Stacked Out, ", target_extracted_confidences_tensor)

          if FLAGS.minimax:
            normalized_ensemble_weights_gpu = ensemble_weights_module_gpu()
            detached_norm_ensemble_weights_gpu = normalized_ensemble_weights_gpu.detach()
            if FLAGS.verbose: print("After Max Step, ", normalized_ensemble_weights_gpu)
          else:
            detached_norm_ensemble_weights_gpu = norm_prior_ensemble_weights_gpu
          detection_loss_gpu = (torch.matmul(detached_norm_ensemble_weights_gpu, target_extracted_confidences_tensor))\
                                   / n_mini_batches
          printability_loss_gpu = (FLAGS.lambda_nps * nps_gpu(adv_patch_gpu)) / n_mini_batches
          patch_variation_loss_gpu = (FLAGS.lambda_tv * tv_gpu(adv_patch_gpu)) / n_mini_batches
          total_loss_gpu = detection_loss_gpu + printability_loss_gpu + patch_variation_loss_gpu

          # GRADIENT UPDATE
          total_loss_gpu.backward()

          # ACCUMULATE BATCH STATISTICS
          batch_ensemble_weights = detached_norm_ensemble_weights_gpu.cpu().numpy()
          batch_total_loss_sum += total_loss_gpu.detach().cpu().numpy()
          batch_detection_loss_sum += detection_loss_gpu.detach().cpu().numpy()
          batch_printability_loss_sum += printability_loss_gpu.detach().cpu().numpy()
          batch_patch_variation_loss_sum += patch_variation_loss_gpu.detach().cpu().numpy()

        # MIN WEIGHT UPDATE
        patch_optimizer.step()
        patch_optimizer.zero_grad()

        # BATCH LOGGING
        if FLAGS.tensorboard_epoch:
          epoch_total_loss_sum += batch_total_loss_sum
          epoch_weighted_detection_loss_sum += batch_detection_loss_sum
          epoch_weighted_printability_loss_sum += batch_printability_loss_sum
          epoch_weighted_patch_variation_loss_sum += batch_patch_variation_loss_sum
          for target_idx, target_name in enumerate(target_extracted_confidences_gpu_dict):
            epoch_unweighted_detector_loss_sum[target_name] += batch_extracted_confidence[target_name]
            epoch_ensemble_weights_sum[target_name] += batch_ensemble_weights[target_idx]

        # write to log
        if FLAGS.tensorboard_batch:
          batch_iteration = iterations_per_epoch * epoch + nth_batch
          tensorboard_writer.add_scalar('batch/total_loss', batch_total_loss_sum, batch_iteration)
          tensorboard_writer.add_scalar('batch/detection_loss', batch_detection_loss_sum, batch_iteration)
          tensorboard_writer.add_scalar('batch/printability_loss', batch_printability_loss_sum, batch_iteration)
          tensorboard_writer.add_scalar('batch/patch_variation_loss', batch_patch_variation_loss_sum, batch_iteration)
          for target_idx, target_name in enumerate(target_extracted_confidences_gpu_dict):
            tensorboard_writer.add_scalar('batch/%s_unweighted_loss'%target_name,
                                          batch_extracted_confidence[target_name], batch_iteration)
            tensorboard_writer.add_scalar('batch/%s_ensemble_weight'%target_name,
                                          batch_ensemble_weights[target_idx], batch_iteration)
      
      
    # EPOCH LOGGING & LR SCHEDULING
    epoch_total_loss_mean = epoch_total_loss_sum/iterations_per_epoch
    patch_lr_scheduler.step(epoch_total_loss_mean)
    adv_patch_cpu_tensor = adv_patch_gpu.detach().cpu()
    im = transforms.ToPILImage('RGB')(adv_patch_cpu_tensor)
    im.save(_patch_path(), "PNG")
    if FLAGS.tensorboard_epoch:
      # calculate statistics
      epoch_weighted_detection_loss_mean = epoch_weighted_detection_loss_sum/iterations_per_epoch
      epoch_weighted_printability_loss_mean = epoch_weighted_printability_loss_sum/iterations_per_epoch
      epoch_weighted_patch_variation_loss_mean = epoch_weighted_patch_variation_loss_sum/iterations_per_epoch
      # write to log
      tensorboard_writer.add_scalar('epoch/total_loss_mean', epoch_total_loss_mean, epoch)
      tensorboard_writer.add_scalar('epoch/detection_loss_mean', epoch_weighted_detection_loss_mean, epoch)
      tensorboard_writer.add_scalar('epoch/printability_loss_mean', epoch_weighted_printability_loss_mean, epoch)
      tensorboard_writer.add_scalar('epoch/patch_variation_loss_mean', epoch_weighted_patch_variation_loss_mean, epoch)
      for target_name in epoch_unweighted_detector_loss_sum:
        epoch_unweighted_detector_loss_mean = epoch_unweighted_detector_loss_sum[target_name] / iterations_per_epoch
        epoch_detector_ensemble_weight_mean = epoch_ensemble_weights_sum[target_name] / iterations_per_epoch
        tensorboard_writer.add_scalar('epoch/%s_unweighted_loss_mean'%target_name, epoch_unweighted_detector_loss_mean,
                                      epoch)
        tensorboard_writer.add_scalar('epoch/%s_ensemble_weight_mean'%target_name, epoch_detector_ensemble_weight_mean,
                                      epoch)
      tensorboard_writer.add_image('patch', adv_patch_cpu_tensor.numpy(), epoch) # tensorboard colors are buggy
    if FLAGS.verbose: print('EPOCH LOSS: %.3f\n'%epoch_total_loss_mean)
    # INTERVAL METRIC REPORTING
    if FLAGS.tune_tracking_interval != 0 and epoch % FLAGS.tune_tracking_interval == 0:
      save_checkpoint(epoch, patch_module_gpu, patch_optimizer, patch_lr_scheduler,
                      ensemble_weights_module_gpu, ensemble_weights_optimizer)
      del patch_module_gpu, patch_optimizer, patch_lr_scheduler, ensemble_weights_module_gpu, ensemble_weights_optimizer
      torch.cuda.empty_cache()
      metric = generate_statistics_and_scalar_metric()
      torch.cuda.empty_cache()
      track.log(worst_case_iou=metric, done=(epoch == (FLAGS.n_epochs - 1)), training_iteration=epoch)
      patch_module_gpu, patch_optimizer, patch_lr_scheduler, ensemble_weights_module_gpu, ensemble_weights_optimizer \
          = allocate_memory_for_stateful_components(cuda_device_id, target_prior_weight)
      _ = load_checkpoint_and_get_epoch(patch_module_gpu, patch_optimizer, patch_lr_scheduler,
                                        ensemble_weights_module_gpu, ensemble_weights_optimizer)



def get_checkpoint_filepath():
  return os.path.join(FLAGS.logdir, 'checkpoint.pth.tar')


def save_checkpoint(epoch:int, patch_module:torch.nn.Module, min_optimizer, min_scheduler,
                    ensemble_weight_module:torch.nn.Module=None, max_optimizer=None):
  state = {'epoch': epoch + 1, 'patch': patch_module.state_dict(), 'min_optimizer': min_optimizer.state_dict(),
           'min_scheduler': min_scheduler.state_dict()}
  if max_optimizer:
    state['max_optimizer'] = max_optimizer.state_dict()
    state['ensemble_weight'] = ensemble_weight_module.state_dict()
  torch.save(state, get_checkpoint_filepath())


def load_checkpoint_and_get_epoch(patch_module:torch.nn.Module, min_optimizer, min_scheduler,
                                  ensemble_weight_module:torch.nn.Module=None, max_optimizer=None) -> int:
  checkpoint = torch.load(get_checkpoint_filepath())
  patch_module.load_state_dict(checkpoint['patch'])
  min_optimizer.load_state_dict(checkpoint['min_optimizer'])
  min_scheduler.load_state_dict(checkpoint['min_scheduler'])
  if max_optimizer:
    max_optimizer.load_state_dict(checkpoint['max_optimizer'])
    ensemble_weight_module.load_state_dict(checkpoint['ensemble_weight'])
  return checkpoint['epoch']


def continue_from_checkpoint_if_exists_and_get_epoch(patch_module:torch.nn.Module, min_optimizer, scheduler,
                                                     ensemble_weight_module:torch.nn.Module=None, max_optimizer=None):
  if os.path.isfile(get_checkpoint_filepath()):
    current_epoch = load_checkpoint_and_get_epoch(patch_module, min_optimizer, scheduler,
                                                  ensemble_weight_module, max_optimizer)
  else:
    current_epoch = 0
  return current_epoch


def allocate_memory_for_stateful_components(cuda_device_id:int, target_prior_weight:dict):
  patch_module_gpu = SquarePatch(patch_size=FLAGS.patch_square_length, typ=FLAGS.start_patch).cuda(cuda_device_id)
  patch_optimizer = torch.optim.Adam(patch_module_gpu.parameters(), lr=FLAGS.lr)
  patch_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patch_optimizer, 'min', patience=FLAGS.plateau_patience)

  ensemble_weights_module_gpu = EnsembleWeights(target_prior_weight).cuda(cuda_device_id)
  max_optimizer_class_init_func = {
    "adam": lambda: torch.optim.Adam(ensemble_weights_module_gpu.parameters(), lr=FLAGS.max_lr),
    "sgd": lambda: torch.optim.SGD(ensemble_weights_module_gpu.parameters(), lr=FLAGS.max_lr)
  }
  ensemble_weights_optimizer = max_optimizer_class_init_func[FLAGS.max_optim]() if FLAGS.minimax else None

  return patch_module_gpu, patch_optimizer, patch_lr_scheduler, ensemble_weights_module_gpu, ensemble_weights_optimizer


def worst_case_iou(detector_statistics, evaluated_detector_names) -> float:
  selected_rows = detector_statistics['target'].apply(lambda el: el in evaluated_detector_names)
  subset = detector_statistics[selected_rows]
  return subset['patch_grand_iou'].max()


def generate_statistics_and_scalar_metric() -> float:
  detector_statistics = test_on_all_detectors(FLAGS.inria_test_dir, _patch_path())
  evaluated_detector_names = []
  flags_dict = FLAGS.flag_values_dict()
  for detector_name in SUPPORTED_TEST_DETECTORS:
    key = "eval_%s" % detector_name
    if key in flags_dict:
      if flags_dict[key]:
        evaluated_detector_names.append(detector_name)
  metric = worst_case_iou(detector_statistics, evaluated_detector_names)
  metric_path = os.path.join(FLAGS.logdir, "metric.txt")
  textfile = open(metric_path, 'w+')
  textfile.write(f'{metric}\n')
  textfile.close()
  if FLAGS.verbose: print('Metric saved to %s\n' % metric_path)
  return metric


def _patch_path():
  return os.path.join(FLAGS.logdir, "patch.png")


def main(argv):
  train()
  if FLAGS.tune_tracking_interval is not 0:
    torch.cuda.empty_cache()
    _ = generate_statistics_and_scalar_metric()


if __name__ == '__main__':
  app.run(main)
