# globally define cli flags
# usage: python app.py --flag_name=value
from absl import flags
from datetime import datetime

from detectors_manager import SUPPORTED_TRAIN_DETECTORS
from test_patch import SUPPORTED_TEST_DETECTORS

FLAGS = flags.FLAGS

# training settings/hyperparameters
flags.DEFINE_integer(name="n_epochs", default=100, help="number of epochs to iterate over training data")
flags.DEFINE_float(name="lr", default=0.03, help="learning rate")
flags.DEFINE_integer(name="bs", default=8, help="batch size")
flags.DEFINE_integer(name="plateau_patience", default=8, help="max number of updates without decrease in loss "
                                                              "or learning rate")
for detector_name in SUPPORTED_TRAIN_DETECTORS:
  flags.DEFINE_integer(name="train_%s"%detector_name, default=0,
                       lower_bound=0, upper_bound=SUPPORTED_TRAIN_DETECTORS[detector_name],
                       help="whether to train patch against %s, 0 for off, "
                            "other positive number for other setting"%detector_name)
  if SUPPORTED_TRAIN_DETECTORS[detector_name] == 3:
    flags.DEFINE_float(name="%s_object_weight"%detector_name, default=0.9, lower_bound=0, upper_bound=1,
                       help="how much is object confidence weighted relative to class confidence "
                            "for %s"%detector_name)
flags.DEFINE_string(name="confidence_processor", default="map_max", help="choice of confidence extraction function")
flags.DEFINE_boolean(name="activate_logits", default=False, help="whether to use probabilities instead of logits when "
                                                                 "extracting detection confidence")

# physical realizability parameters
flags.DEFINE_float(name="lambda_nps", default=0.01, help="multiplier for non printability score")
flags.DEFINE_float(name="lambda_tv", default=2.5, help="multiplier for patch total variation")

# patch settings
flags.DEFINE_enum(name="start_patch", default="grey", enum_values=["grey", "random"], help="start with grey or "
                                                                                           "random patch")
flags.DEFINE_integer(name="patch_square_length", default=300, help="side length of the square patch")

# debug
flags.DEFINE_boolean(name="debug_autograd", default=False, help="whether to use extra computation on debugging autograd")
flags.DEFINE_boolean(name="debug_device", default=False, help="whether to use extra computation on "
                                                              "debugging tensor devices")

# testing settings
for detector_name in SUPPORTED_TEST_DETECTORS:
  flags.DEFINE_boolean(name="test_%s"%detector_name, default=False,
                       help="whether to test patch against %s"%detector_name)

# data settings
flags.DEFINE_integer(name="max_labs", default=14, help="maximum number of bounding boxes to load in per image, "
                                                       "decides label tensor size and processing speed")
flags.DEFINE_string(name="inria_dir", default="inria/Train/pos", help="directory storing the people pics for INRIA")
flags.DEFINE_integer(name="init_size", default=608, help="initial side length of loaded image")
flags.DEFINE_integer(name="num_workers", default=8, help="number of threaded workers for loading data")
flags.DEFINE_string(name="printable_vals_filepath", default="non_printability/30values.txt",
                    help="txt file containing vector of printable rgb values for calculation non printability score")

# logging settings
_init_time = datetime.now()
flags.DEFINE_string(name="logdir",
                    default="logs/%s%s"%(_init_time.astimezone().tzinfo.tzname(None),
                                         _init_time.strftime('%Y%m%d_%H_%M_%S_%f')),
                    help="directory to store logs, images, and statistics")
