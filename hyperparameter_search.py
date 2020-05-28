import os
import sys
import argparse
import json
from datetime import datetime

import ConfigSpace as CS

from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from train_test_patch_one_gpu import train

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
  flags = f'python3 ../../train_test_patch_one_gpu.py {standard_flags}'
  for i in config:
    flags += f' --{i}={str(config[i])}'
  argv = flags.split()[1:]
  FLAGS(argv)
  train()

# Define logdir file, create it if does not exist
_init_time = datetime.now()
logdir = f"logs_hpo_{_init_time.astimezone().tzinfo.tzname(None)+_init_time.strftime('%Y%m%d_%H_%M_%S_%f')}"

if not os.path.exists(logdir):
  os.makedirs(logdir)
if setting_list['redirect_stdout'] == 'True':
  sys.stdout = open(os.path.join(logdir, "stdout.txt"), "w")
    
# Load JSON config file
with open(config_file) as config_file:
    data = json.load(config_file)

# Extract settings + hyperparameter config
config = data['hyperparameter_config_space']
setting_list = data['settings']

# Get number of samples and mini batch size
n_samples = setting_list['n_samples']
mini_batch_size = setting_list['mini_batch_size']

# Check if early stopping is enabled. If it is, include tracking interval in flags. Also calculate n_epochs
if setting_list['early_stopping'] == 'True':
  tracking_interval = setting_list['tracking_interval']
  standard_flags+=f" --tune_tracking_interval={tracking_interval}"
  n_epochs = int(tracking_interval)*int(setting_list['n_tracking_intervals'])
else:
  n_epochs = int(setting_list['n_epochs'])

# Add n_epochs and mini batch size to flags
standard_flags+=f' --n_epochs={n_epochs} --mini_bs={mini_batch_size}' 

# Add constant hyperparameters to flags
for constant,value in config['constants'].items():
  standard_flags+=f' --{constant}={value}' 
  
# Extract hyperparameters from JSON file and add to configuration space. Also account for any constraints.
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

# Run hyperparameter optimization
experiment_metrics = dict(metric="worst_case_iou", mode="min")
bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration",max_t=n_epochs,**experiment_metrics)
bohb_search = TuneBOHB(config_space, **experiment_metrics)

analysis = tune.run(train_one_gpu_early_stopping,
    name=logdir,
    scheduler=bohb_hyperband,
    search_alg=bohb_search,
    num_samples=n_samples, resources_per_trial={"gpu":1}, local_dir="./")
    
print("Best config: ", analysis.get_best_config(metric="worst_case_iou", mode="min"))

# saves relevant summary data to file under logdir
df = analysis.dataframe()
df.to_csv(os.path.join(logdir, "data.csv"))