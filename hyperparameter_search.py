import os
import sys
import argparse
from datetime import datetime

import ConfigSpace as CS

from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from train_test_patch_one_gpu import train

parser = argparse.ArgumentParser()
parser.add_argument("--nt", type=int, default=20, help="number of trials")
parser.add_argument("--mbs", type=int, default=8, help="minibatch size")
parser.add_argument("--ti", type=int, default=10, help="number of epochs in a tracking interval, "
                                                       "0 to disable interval tracking")
parser.add_argument("--ni", type=int, default=5, help="number of tracking intervals in a session")
parser.add_argument("--ne", type=int, default=50, help="number of epochs in a session, "
                                                       "for when interval tracking is disabled")
parser.add_argument("--rs", type=bool, default=True, help="whether to redirect stdout to logfile")
parser.add_argument("--rsm", type=str, default=None, help="directory to resume session from")

args = parser.parse_args()
mini_batch_size = args.mbs
tracking_interval = args.ti
n_epochs = tracking_interval*args.ni if args.ti != 0 else args.ne

_init_time = datetime.now()
if args.rsm:
  logdir = args.rsm
else:
  logdir = f"logs_hpo_{_init_time.astimezone().tzinfo.tzname(None)+_init_time.strftime('%Y%m%d_%H_%M_%S_%f')}"
  if not os.path.exists(logdir):
    os.makedirs(logdir)

if args.rs:
  sys.stdout = open(os.path.join(logdir, "stdout.txt"), "a")

standard_flags = f'--eval_yolov2=True --eval_ssd=True --eval_yolov3=True ' \
                 f'--n_epochs={n_epochs} --mini_bs={mini_batch_size} ' \
                 f'--inria_train_dir=../../inria/Train/pos --printable_vals_filepath=../../non_printability/30values.txt ' \
                 f'--inria_test_dir=../../inria/Test/pos --logdir=logs --yolov2_cfg_file=../../cfg/yolov2.cfg ' \
                 f'--yolov2_weight_file=../../weights/yolov2.weights --yolov3_cfg_file=../../implementations/yolov3/config/yolov3.cfg ' \
                 f'--yolov3_weight_file=../../implementations/yolov3/weights/yolov3.weights ' \
                 f'--ssd_weight_file=../../implementations/ssd/models/vgg16-ssd-mp-0_7726.pth ' \
                 f'--example_patch_file=../../saved_patches/perry_08-26_500_epochs.jpg ' \
                 f'--tensorboard_epoch=False'


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
    
    
config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("lr", lower=0.00001, upper=.1))
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("num_mini", lower=1, upper=20))
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("plateau_patience", lower=1, upper=n_epochs))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("activate_logits", choices=["True","False"]))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("confidence_processor", choices=["avg", "max", "det_max", "det_avg", "det_max_avg"]))

min_max_hp = CS.CategoricalHyperparameter("minimax", choices=["True","False"])
config_space.add_hyperparameter(min_max_hp)

minimax_gamma_hp = CS.UniformFloatHyperparameter("minimax_gamma", lower=0.00001, upper=100)
config_space.add_hyperparameter(minimax_gamma_hp)
config_space.add_condition(CS.EqualsCondition(minimax_gamma_hp, min_max_hp, "True"))

max_optim_hp = CS.CategoricalHyperparameter("max_optim", choices=["sgd","adam"])
config_space.add_hyperparameter(max_optim_hp)
config_space.add_condition(CS.EqualsCondition(max_optim_hp, min_max_hp, "True"))

max_lr_hp = CS.UniformFloatHyperparameter("max_lr", lower=0.00001, upper=.1)
config_space.add_hyperparameter(max_lr_hp)
config_space.add_condition(CS.EqualsCondition(max_lr_hp, min_max_hp, "True"))


config_space.add_hyperparameter(CS.CategoricalHyperparameter("start_patch", choices=["grey","random"]))

train_yolov2_hp = CS.CategoricalHyperparameter("train_yolov2", choices=["1","2","3"])
config_space.add_hyperparameter(train_yolov2_hp)

train_yolov3_hp = CS.CategoricalHyperparameter("train_yolov3", choices=["1","2","3"])
config_space.add_hyperparameter(train_yolov3_hp)

config_space.add_hyperparameter(CS.CategoricalHyperparameter("train_ssd", choices=["1"]))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("yolov2_prior_weight", lower=-10, upper=10))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("ssd_prior_weight", lower=-10, upper=10))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("yolov3_prior_weight", lower=-10, upper=10))

yolov2_object_weight_hp = CS.UniformFloatHyperparameter("yolov2_object_weight", lower=0.00001, upper=1)
config_space.add_hyperparameter(yolov2_object_weight_hp)
config_space.add_condition(CS.EqualsCondition(yolov2_object_weight_hp, train_yolov2_hp, "3"))

yolov3_object_weight_hp = CS.UniformFloatHyperparameter("yolov3_object_weight", lower=0.00001, upper=1)
config_space.add_hyperparameter(yolov3_object_weight_hp)
config_space.add_condition(CS.EqualsCondition(yolov3_object_weight_hp, train_yolov3_hp, "3"))

experiment_metrics = dict(metric="worst_case_iou", mode="min")
bohb_hyperband = HyperBandForBOHB(time_attr="reporting_interval",max_t=n_epochs,**experiment_metrics)
bohb_search = TuneBOHB(config_space, **experiment_metrics)

analysis = tune.run(train_one_gpu_early_stopping,
    name=logdir,
    scheduler=bohb_hyperband,
    search_alg=bohb_search,
    resume=(args.rsm is not None),
    num_samples=args.nt, resources_per_trial={"gpu":1}, local_dir="./")
print("Best config: ", analysis.get_best_config(metric="worst_case_iou", mode="min"))
# saves relevant summary data to file under logdir
df = analysis.dataframe()
df.to_csv(os.path.join(logdir, "data.csv"))
