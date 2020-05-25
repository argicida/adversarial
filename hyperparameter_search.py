import os
from datetime import datetime

import ConfigSpace as CS

from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from train_test_patch_one_gpu import train

tracking_interval = 10
standard_flags = f'--eval_yolov2=True --eval_ssd=True --eval_yolov3=True ' \
                 f'--inria_train_dir=../../inria/Train/pos --printable_vals_filepath=../../non_printability/30values.txt ' \
                 f'--inria_test_dir=../../inria/Test/pos --logdir=logs --yolov2_cfg_file=../../cfg/yolov2.cfg ' \
                 f'--yolov2_weight_file=../../weights/yolov2.weights --yolov3_cfg_file=../../implementations/yolov3/config/yolov3.cfg ' \
                 f'--yolov3_weight_file=../../implementations/yolov3/weights/yolov3.weights ' \
                 f'--ssd_weight_file=../../implementations/ssd/models/vgg16-ssd-mp-0_7726.pth ' \
                 f'--example_patch_file=../../saved_patches/perry_08-26_500_epochs.jpg ' \
                 f'--tensorboard_epoch=False --train_ssd=1'

def train_one_gpu(config):
    flags = f'python3 ../../train_test_patch_one_gpu.py {standard_flags}'
    for i in config:
      flags+=f' --{i}={str(config[i])}' 
    os.system(flags)
    if os.path.exists("logs/metric.txt"):
      textfile = open("logs/metric.txt", 'r')
      metric = float(textfile.readline())
      textfile.close()
      #os.remove("logs/metric.txt")
      tune.track.log(worst_case_iou=metric, done=True)
    else:
      print("Trial Failed!")


def train_one_gpu_early_stopping(config):
  from cli_config import FLAGS
  FLAGS.unparse_flags()
  flags = f'python3 ../../train_test_patch_one_gpu.py {standard_flags} --tune_tracking_interval={tracking_interval}'
  for i in config:
    flags += f' --{i}={str(config[i])}'
  argv = flags.split()[1:]
  FLAGS(argv)
  train()
  # if os.path.exists("logs/metric.txt"):
    # textfile = open("logs/metric.txt", 'r')
    # metric = float(textfile.readline())
    # textfile.close()
    # os.remove("logs/metric.txt")
    # tune.track.log(worst_case_iou=metric, done=True)
  # else:
    # print("Trial Failed!")
    
with open(config_file) as config_file:
    data = json.load(config_file)

n_samples = data['hyperparameter_search_settings']['num_samples']
config = data['hyperparameter_config_space']
mini_batch_size = config['constants']['mini_batch_size']
tracking_interval = int(config['constants']['tracking_interval'])
n_epochs = tracking_interval * 5
standard_flags+=f' --n_epochs={n_epochs} --mini_bs={mini_batch_size}' 

config_space = CS.ConfigurationSpace()
constraints = {}
for name,settings in config['search_space'].items():
  hp_type = settings['type']
  if hp_type == 'UF':
    hp = CS.UniformFloatHyperparameter(name, lower=float(settings['lower']), upper=float(settings['upper']))
  elif hp_type == 'UI':
    hp = CS.UniformIntegerHyperparameter(name, lower=float(settings['lower']), upper=float(settings['upper']))
  elif hp_type == 'C':
    hp = CS.CategoricalHyperparameter(name, choices=settings['options'].split(','))
  config_space.add_hyperparameter(hp)
  if name == 'train_yolov2' or name == 'train_yolov3' or name == 'minimax':
    constraints[name] = hp
  if 'condition' in settings:
    conditions = settings['condition'].split(',')
    config_space.add_condition(CS.EqualsCondition(hp, constraints[conditions[0]], conditions[1]))

experiment_metrics = dict(metric="worst_case_iou", mode="min")
bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration",max_t=n_epochs,**experiment_metrics)
bohb_search = TuneBOHB(config_space, **experiment_metrics)

_init_time = datetime.now()
analysis = tune.run(train_one_gpu_early_stopping,
    name=f"logs_hp_optim_{_init_time.astimezone().tzinfo.tzname(None)+_init_time.strftime('%Y%m%d_%H_%M_%S_%f')}",
    scheduler=bohb_hyperband,
    search_alg=bohb_search,
    num_samples=n_samples, resources_per_trial={"gpu":1}, local_dir="./")
print("Best config: ", analysis.get_best_config(metric="worst_case_iou", mode="min"))