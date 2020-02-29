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
for detector_name in SUPPORTED_TRAIN_DETECTORS:
  flags.DEFINE_integer(name="train_%s"%detector_name, default=0,
                       lower_bound=0, upper_bound=SUPPORTED_TRAIN_DETECTORS[detector_name],
                       help="whether to train patch against %s, 0 for off, "
                            "other positive number for other setting"%detector_name)
flags.DEFINE_boolean(name="gt_cost", default=True, help="whether to utilize ground truth bounding boxes"
                                                        "when extracting detection confidence to minimize")
flags.DEFINE_boolean(name="logits_cost", default=True, help="whether to use logits instead of probabilities when"
                                                            "extracting detection confidence")
flags.DEFINE_enum(name="start_patch", default="grey", enum_values=["grey", "random"], help="start with grey or "
                                                                                           "random patch")

# testing settings
for detector_name in SUPPORTED_TEST_DETECTORS:
  flags.DEFINE_boolean(name="test_%s"%detector_name, default=False,
                       help="whether to test patch against %s"%detector_name)

# data settings
flags.DEFINE_integer(name="max_labs", default=14, help="maximum number of bounding boxes to load in per image, "
                                                       "decides label tensor size and processing speed")

# logging settings
_init_time = datetime.now()
flags.DEFINE_string(name="log_dir",
                    default="logs/%s%s"%(_init_time.astimezone().tzinfo.tzname(None),
                                         _init_time.strftime('%Y%m%d_%H_%M_%S_%f')),
                    help="directory to store logs, images, and statistics")
