# globally define cli flags
# usage: python app.py --flag_name=value
from absl import flags
from datetime import datetime

from detectors_manager import SUPPORTED_TRAIN_DETECTORS
from test_patch import SUPPORTED_TEST_DETECTORS
from detections_map_processors import uninformed_detections_processor_choices, informed_detections_processor_choices,\
    processor_choices

FLAGS = flags.FLAGS

# training settings/hyperparameters
flags.DEFINE_integer(name="n_epochs", default=100, lower_bound=0, help="number of epochs to iterate over training data")
flags.DEFINE_float(name="lr", default=0.03, lower_bound=0, help="learning rate")
flags.DEFINE_integer(name="bs", default=8, lower_bound=0, help="batch size")
flags.DEFINE_integer(name="mini_bs", default=8, lower_bound=0, help="mini batch size")
flags.DEFINE_integer(name="plateau_patience", default=8, lower_bound=0, help="max number of updates without decrease in "
                                                                             "loss or learning rate")
flags.DEFINE_boolean(name="activate_logits", default=True, help="whether to use probabilities instead of logits when "
                                                                "extracting detection confidence")
for detector_name in SUPPORTED_TRAIN_DETECTORS:
  flags.DEFINE_integer(name="train_%s"%detector_name, default=0,
                       lower_bound=0, upper_bound=SUPPORTED_TRAIN_DETECTORS[detector_name],
                       help="whether to train patch against %s, 0 for off, "
                            "other positive number for other setting"%detector_name)
  flags.DEFINE_float(name="%s_prior_weight"%detector_name, default=0, lower_bound=-100, upper_bound=100,
                     help="the pre-softmax weight assigned to the extracted output of the detector")
  if SUPPORTED_TRAIN_DETECTORS[detector_name] == 3:
    flags.DEFINE_float(name="%s_object_weight"%detector_name, default=0.9, lower_bound=0, upper_bound=1,
                       help="how much is object confidence weighted relative to class confidence "
                            "(which is weighted 1 - object confidence) "
                            "for %s"%detector_name)
flags.DEFINE_enum(name="confidence_processor", default="max",
                  enum_values=uninformed_detections_processor_choices()+informed_detections_processor_choices(),
                  help="choice of confidence extraction function, %s"%str(processor_choices()))

flags.DEFINE_boolean(name="minimax", default=False, help="whether to use minimax formulation to minimize worst case loss")
flags.DEFINE_float(name="minimax_gamma", default=1, lower_bound=0, help="at 0, ensemble prior has no importance. at "
                                                                        "infinity, minimax is not doing anything other "
                                                                        "than using the prior ensemble weights")
flags.DEFINE_float(name="max_lr", default=0.03, lower_bound=0, help="learning rate for ensemble weights update")
flags.DEFINE_enum(name="max_optim", default="adam", enum_values=["sgd", "adam"], help="choice of optimizer for max step")

# physical realizability parameters
flags.DEFINE_float(name="lambda_nps", default=0.01, help="multiplier for non printability score")
flags.DEFINE_float(name="lambda_tv", default=2.5, help="multiplier for patch total variation")

# patch settings
flags.DEFINE_enum(name="start_patch", default="grey", enum_values=["grey", "random"], help="start with grey or "
                                                                                           "random patch")
flags.DEFINE_integer(name="patch_square_length", default=300, help="side length of the square patch")

# debug detectors and patching
flags.DEFINE_boolean(name="debug_autograd", default=False, help="whether to use extra computation on debugging autograd")
flags.DEFINE_boolean(name="debug_device", default=False, help="whether to use extra computation on "
                                                              "debugging tensor devices")
flags.DEFINE_boolean(name="debug_coords", default=False, help="whether to use extra computation on "
                                                              "debugging detector coordinates output")
flags.DEFINE_boolean(name="plot_patches", default=False, help="whether to use matplotlib to display patched images")

# patch evaluation settings
for detector_name in SUPPORTED_TEST_DETECTORS:
  flags.DEFINE_boolean(name="eval_%s"%detector_name, default=False,
                       help="whether to evaluate patch against %s"%detector_name)

# data settings
flags.DEFINE_integer(name="max_labs", default=14, help="maximum number of bounding boxes to load in per image, "
                                                       "decides label tensor size and processing speed")
flags.DEFINE_string(name="inria_train_dir", default="inria/Train/pos",
                    help="directory storing the people pics for INRIA train set")
flags.DEFINE_string(name="inria_test_dir", default="inria/Test/pos",
                    help="directory storing the people pics for INRIA test set")
flags.DEFINE_integer(name="init_size", default=608, help="initial side length of loaded image")
flags.DEFINE_integer(name="num_workers", default=8, help="number of threaded workers for loading data")
flags.DEFINE_string(name="printable_vals_filepath", default="non_printability/30values.txt",
                    help="txt file containing vector of printable rgb values for calculation non printability score")
flags.DEFINE_string(name="yolov2_cfg_file", default="cfg/yolov2.cfg",
                    help="directory for yolov2 cfg file")
flags.DEFINE_string(name="yolov2_weight_file", default="weights/yolov2.weights",
                    help="directory for yolov2 weight file")
flags.DEFINE_string(name="yolov3_cfg_file", default="./implementations/yolov3/config/yolov3.cfg",
                    help="directory for yolov3 cfg file")
flags.DEFINE_string(name="yolov3_weight_file", default="./implementations/yolov3/weights/yolov3.weights",
                    help="directory for yolov3 weight file")
flags.DEFINE_string(name="ssd_weight_file", default="./implementations/ssd/models/vgg16-ssd-mp-0_7726.pth",
                    help="directory for ssd weight file")
flags.DEFINE_string(name="example_patch_file", default="legacy_patches/perry_08-26_500_epochs.jpg",
                    help="directory for example patch file")

# logging settings
_init_time = datetime.now()
flags.DEFINE_string(name="logdir",
                    default="logs/%s%s"%(_init_time.astimezone().tzinfo.tzname(None),
                                         _init_time.strftime('%Y%m%d_%H_%M_%S_%f')),
                    help="directory to store logs, images, and statistics")
flags.DEFINE_boolean(name="tensorboard_batch", default=False, help="whether to log batch statistics to tensorboard;"
                                                                   "potential to take up tons of storage")
flags.DEFINE_boolean(name="tensorboard_epoch", default=True, help="whether to log epoch statistics to tensorboard;"
                                                                  "potential to take up a lot of storage")
flags.DEFINE_boolean(name="verbose", default=False, help="whether to print program status to stdout;"
                                                         "potential to take up a lot of storage")
flags.DEFINE_integer(name="tune_tracking_interval", default=0, lower_bound=0,
                     help="number of epochs for metric reporting intervals, disabled at 0")
