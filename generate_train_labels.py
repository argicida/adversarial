from cli_config import FLAGS
from test_patch import SUPPORTED_TEST_DETECTORS, DETECTOR_LOADERS_N_WRAPPERS, DETECTOR_INPUT_SIZES
from PIL import Image
import os
import fnmatch


def main():
  train_data_dir = "inria/Train/pos"
  flags_dict = FLAGS.flag_values_dict()
  detectors = {}
  detector_wrappers = {}
  for candidate in SUPPORTED_TEST_DETECTORS:
    if flags_dict['test_%s'%candidate]:
      loader_fn, wrapper_fn = DETECTOR_LOADERS_N_WRAPPERS[candidate]
      cuda_device_id = 0
      detectors[candidate] = loader_fn(cuda_device_id)
      detector_wrappers[wrapper_fn] = wrapper_fn
  img_names = fnmatch.filter(os.listdir(train_data_dir), '*.png') + fnmatch.filter(os.listdir(train_data_dir), '*.jpg')
  print("Train Set Size: %s"%len(img_names))
  for img_name in img_names:
    img = Image.open(os.path.join(train_data_dir, img_name)).convert('RGB')
    for detector_name in detectors:
      lab_path = os.path.join(train_data_dir, "%s%s"%(detector_name, "-labels"),
                              img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
      input_size = DETECTOR_INPUT_SIZES[detector_name]
      detector = detectors[detector_name]
      wrapper_fn = detector_wrappers[detector_name]
      resized_img = img.resize((input_size, input_size))
      df = wrapper_fn(resized_img, detector, input_size, input_size) # ["x0", "y0", "w", 'h', 'human']
      textfile = open(lab_path, 'w+')
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
  main()

