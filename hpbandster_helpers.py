from hpbandster.core.worker import Worker
from datetime import datetime
import os
import sys
import json

import ConfigSpace as CS


def get_configspace_and_basecmd(config_filepath):
  _init_time = datetime.now()
  time = _init_time.astimezone().tzinfo.tzname(None) + _init_time.strftime('%Y%m%d_%H_%M_%S_%f')

  with open(config_filepath) as config_file:
    data = json.load(config_file)
    # makes a copy of it in the session log
    with open(f"config_files/hpo_{time}_config.json", 'w') as copy:
      json.dump(data, copy)
  setting_list = data['settings']

  # n_samples = int(setting_list['n_samples'])
  mini_batch_size = setting_list['mini_batch_size']
  # base_cmd = f'python3 ../../train_test_patch_one_gpu.py --eval_yolov2=True --eval_ssd=True --eval_yolov3=True ' \
  #            f'--inria_train_dir=../../inria/Train/pos --printable_vals_filepath=../../non_printability/30values.txt ' \
  #            f'--inria_test_dir=../../inria/Test/pos --logdir=logs ' \
  #            f'--yolov2_cfg_file=../../cfg/yolov2.cfg --yolov2_weight_file=../../weights/yolov2.weights ' \
  #            f'--yolov3_cfg_file=../../implementations/yolov3/config/yolov3.cfg --yolov3_weight_file=../../implementations/yolov3/weights/yolov3.weights ' \
  #            f'--ssd_weight_file=../../implementations/ssd/models/vgg16-ssd-mp-0_7726.pth ' \
  #            f'--example_patch_file=../../saved_patches/perry_08-26_500_epochs.jpg ' \
  #            f'--tensorboard_epoch=False ' \
  #            f'--train_ssd=1 ' \
  #            f'--mini_bs={mini_batch_size}'
  
  base_cmd = f'python3 ./train_test_patch_one_gpu.py --eval_yolov2=True --eval_ssd=True --eval_yolov3=True ' \
             f'--inria_train_dir=./inria/Train/pos --printable_vals_filepath=./non_printability/30values.txt ' \
             f'--inria_test_dir=./inria/Test/pos --logdir=logs ' \
             f'--yolov2_cfg_file=./cfg/yolov2.cfg --yolov2_weight_file=./weights/yolov2.weights ' \
             f'--yolov3_cfg_file=./implementations/yolov3/config/yolov3.cfg --yolov3_weight_file=./implementations/yolov3/weights/yolov3.weights ' \
             f'--ssd_weight_file=./implementations/ssd/models/vgg16-ssd-mp-0_7726.pth ' \
             f'--example_patch_file=./saved_patches/perry_08-26_500_epochs.jpg ' \
             f'--tensorboard_epoch=False ' \
             f'--train_ssd=1 ' \
             f'--mini_bs={mini_batch_size} '\
             f'--verbose=True'

  config = data['hyperparameter_config_space']
  config_space = CS.ConfigurationSpace()
  constraints = {}
  for constant, value in config['constants'].items():
    base_cmd += f' --{constant}={value}'
  for name, settings in config['search_space'].items():
    hp_type = settings['type']
    if hp_type == 'UF':
      hp = CS.UniformFloatHyperparameter(name, lower=float(settings['lower']), upper=float(settings['upper']))
    elif hp_type == 'UI':
      hp = CS.UniformIntegerHyperparameter(name, lower=int(settings['lower']), upper=int(settings['upper']))
    elif hp_type == 'C':
      hp = CS.CategoricalHyperparameter(name, choices=settings['options'].split(','))
    else:
      raise ValueError(f"Undefined Hyperparameter Type: {hp_type}")
    config_space.add_hyperparameter(hp)
    if name == 'train_yolov2' or name == 'train_yolov3' or name == 'minimax':
      constraints[name] = hp
    if 'condition' in settings:
      conditions = settings['condition'].split(',')
      config_space.add_condition(CS.EqualsCondition(hp, constraints[conditions[0]], conditions[1]))
  return base_cmd, config_space

class PatchWorker(Worker):
  def __init__(self, base_cmd, *args, **kwargs):
    super(PatchWorker, self).__init__(*args, **kwargs)
    self.base_command = base_cmd
    print(f"worker {self.run_id} Initialized")

  def compute(self, config, budget, *args, **kwargs):
    now = datetime.now()
    logdir = f"logs/hpo_run_{now.astimezone().tzinfo.tzname(None) + now.strftime('%Y%m%d_%H_%M_%S_%f')}"
    command = self.base_command
    for i in config:
      command += f' --{i}={str(config[i])}'
    # budget gets rounded for epoch https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_keras_worker.html
    command += f' --n_epochs={int(budget)} --logdir={logdir}'
    print(logdir, "Working Directory", os.getcwd(), "Command", command)
    os.system(command)
    metric_filepath = os.path.join(logdir, 'metric.txt')
    if os.path.exists(metric_filepath):
      with open(metric_filepath, 'r') as textfile:
        metric = float(textfile.readline())
      print(logdir, "Terminated")
      return {
        'loss': metric,
        'info': None
      }
    else:
      print(logdir, "Failed")
      return None
