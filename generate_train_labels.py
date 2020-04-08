from absl import app, flags
from test_patch import SUPPORTED_TEST_DETECTORS, DETECTOR_LOADERS_N_WRAPPERS, DETECTOR_INPUT_SIZES
from PIL import Image
from tqdm import tqdm
import os
import fnmatch

for detector_name in SUPPORTED_TEST_DETECTORS:
  flags.DEFINE_boolean(name=detector_name, default=False,
                       help="whether to generate labels for %s"%detector_name)
flags.DEFINE_string(name="inria_dir", default="inria/Train/pos", help="directory storing the people pics for INRIA")
FLAGS = flags.FLAGS

## USAGE EXAMPLE: python3 generate_train_labels.py --yolov2=True --ssd=True


def main(argv):
  train_data_dir = FLAGS.inria_dir
  flags_dict = FLAGS.flag_values_dict()
  detectors = {}
  detector_wrappers = {}
  for candidate in SUPPORTED_TEST_DETECTORS:
    if flags_dict[candidate]:
      loader_fn, wrapper_fn = DETECTOR_LOADERS_N_WRAPPERS[candidate]
      cuda_device_id = 0
      detectors[candidate] = loader_fn(cuda_device_id)
      detector_wrappers[candidate] = wrapper_fn
      lab_dir = os.path.join(train_data_dir, "%s%s" % (candidate, "-labels"))
      if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
  img_names = fnmatch.filter(os.listdir(train_data_dir), '*.png') + fnmatch.filter(os.listdir(train_data_dir), '*.jpg')
  print("Number of Pics in Train Set: %s"%len(img_names))
  for img_name in tqdm(img_names):
    img = Image.open(os.path.join(train_data_dir, img_name)).convert('RGB')
    txt = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    for detector_id in detectors:
      lab_dir = os.path.join(train_data_dir, "%s%s" % (detector_id, "-labels"))
      lab_path = os.path.join(lab_dir, txt)
      input_size = DETECTOR_INPUT_SIZES[detector_id]
      detector = detectors[detector_id]
      wrapper_fn = detector_wrappers[detector_id]
      resized_img = img.resize((input_size, input_size))
      df = wrapper_fn(resized_img, detector, input_size, input_size) # ["x0", "y0", "w", 'h', 'human']
      textfile = open(lab_path, 'w+')
      #print(detector_id)
      #print(df)
      for index, box in df.iterrows():
        norm_x0 = box['x0'] / input_size
        norm_y0 = box['y0'] / input_size
        norm_width = box['w'] / input_size
        norm_height = box['h'] / input_size
        if box['human']:
          cls_id = 0
          textfile.write(f'{cls_id} {norm_x0 + 0.5*norm_width} {norm_y0 + 0.5*norm_height} {norm_width} {norm_height}\n')
      textfile.close()


if __name__ == "__main__":
  app.run(main)

