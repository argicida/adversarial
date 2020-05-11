import os
from datetime import datetime

import ConfigSpace as CS

from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from train_test_patch_one_gpu import train

tracking_interval = 10
n_epochs = tracking_interval*5
standard_flags = f'--eval_yolov2=True --eval_ssd=True --eval_yolov3=True --n_epochs={n_epochs} --bs=8 ' \
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
  # if os.path.exists("logs/metric.txt"):
    # textfile = open("logs/metric.txt", 'r')
    # metric = float(textfile.readline())
    # textfile.close()
    # os.remove("logs/metric.txt")
    # tune.track.log(worst_case_iou=metric, done=True)
  # else:
    # print("Trial Failed!")
    
    
config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("lr", lower=0.00001, upper=.1))
#config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("bs", lower=1, upper=24))
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
bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration",max_t=n_epochs,**experiment_metrics)
bohb_search = TuneBOHB(config_space, **experiment_metrics)

_init_time = datetime.now()
analysis = tune.run(train_one_gpu_early_stopping,
    name=f"logs_hp_optim_{_init_time.astimezone().tzinfo.tzname(None)+_init_time.strftime('%Y%m%d_%H_%M_%S_%f')}",
    scheduler=bohb_hyperband,
    search_alg=bohb_search,
    num_samples=10, resources_per_trial={"gpu":1}, local_dir="./")
print("Best config: ", analysis.get_best_config(metric="worst_case_iou", mode="min"))
