import os

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import torch

import matplotlib.pyplot as plt

from darknet import Darknet
import utils

from multiprocessing import Process, Queue


class VideoPipe(Pipeline):
  def __init__(self, batch_size, sequence_length, initial_prefetch_size, nchw, num_threads, device_id, data, shuffle):
    super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)
    self.input = ops.VideoReader(device="gpu", normalized=False, filenames=data, sequence_length=sequence_length,
                                 shard_id=0, num_shards=1,
                                 random_shuffle=shuffle, initial_fill=initial_prefetch_size)
    self.transpose = ops.Transpose(device="gpu", perm=[0, 3, 1, 2])
    self.nchw = nchw

  def define_graph(self):
    output = self.input(name="Reader")
    if self.nchw:
      output = self.transpose(output)
    return output

class TorchVideoFramesLoaderIterable():
  # single process frames loading from video using nvidia dali
  def __init__(self, sequence_length, video_directory, nchw=True, num_threads=4, shuffle=False, device_id=0):
    batch_size = 1 # sequence_length will be batching frames, so this will be 1
    initial_prefetch_size=16
    video_files=[video_directory + '/' + f for f in os.listdir(video_directory) if not f.endswith('.txt')]
    self.pipeline = VideoPipe(batch_size=batch_size, sequence_length=sequence_length, initial_prefetch_size=initial_prefetch_size, nchw=nchw, num_threads=num_threads, device_id=device_id, data=video_files, shuffle=shuffle)
    self.pipeline.build()
    self.num_batches = self.pipeline.epoch_size("Reader")
    self.dali_iterator = DALIGenericIterator(self.pipeline, ["data"], self.num_batches, auto_reset=True)

  def __len__(self):
    return int(self.num_batches)

  def __iter__(self):
    return self.dali_iterator.__iter__()


def padded_square_tensor(tensor):
  h, w = tensor.size()[-2:]
  if h > w:
    padded = torch.ones(tensor.size()[0:2] + (h, h))
    padded[:, :, :, :w] = tensor
    del tensor
    return padded
  elif w > h:
    padded = torch.ones(tensor.size()[0:2] + (w, w))
    padded[:, :, :h, :] = tensor
    del tensor
    return padded
  else:
    return tensor


def view_first_image_in_nchw_batch(nchw_pytorch_tensor, title=None):
  # incorrectly displays 9 clones of the first image rather than one big image, but shouldn't impact functionality
  nhwc_batch = nchw_pytorch_tensor.view(nchw_pytorch_tensor.shape[0], nchw_pytorch_tensor.shape[2],
                                        nchw_pytorch_tensor.shape[3], nchw_pytorch_tensor.shape[1]).cpu()
  plt.imshow(nhwc_batch[0])
  if title is not None:
    plt.title(title)
  plt.show()


def nms_and_count_human_boxes(nms_threshold, input_queue:Queue, output_queue:Queue):
  human_count = 0
  while True:
    item = input_queue.get()
    if item == "SIGKILL":
      break
    boxes = utils.nms(item, nms_threshold)
    for box in boxes:
        if box[6] == 0: # yolov2 class person
            human_count += 1
  output_queue.put(human_count)


def main():
  cfgfile = "cfg/yolov2.cfg"
  weightfile = "weights/yolov2.weights"
  darknet_model = Darknet(cfgfile)
  darknet_model.load_weights(weightfile)
  darknet_model = darknet_model.eval().cuda()
  detection_confidence_threshold = 0.5
  nms_threshold = 0.4

  batch_size = 4
  video_directory = "test_videos/"  
  loader_iterable = TorchVideoFramesLoaderIterable(batch_size, video_directory, nchw=True)
  # iterate through all the frames of all the videos in the directory in batch through pytorch tensors
  final_dim = (darknet_model.height, darknet_model.width)

  human_count = 0
  object_count = 0
  num_frames = batch_size * len(loader_iterable)

  for out in loader_iterable:
    batch = torch.squeeze(out[0]['data']).cuda()
    # view_first_image_in_nchw_batch(batch, "batch")
    normalized_batch = torch.div(batch.type(torch.float32), 255).cuda()
    # view_first_image_in_nchw_batch(normalized_batch, "normalized_batch")
    del batch
    square_batch = padded_square_tensor(normalized_batch).cuda()
    # view_first_image_in_nchw_batch(square_batch, "square_batch")
    del normalized_batch
    resized_batch = torch.nn.functional.interpolate(square_batch, size=final_dim, mode='bilinear').cuda()
    # view_first_image_in_nchw_batch(resized_batch, "resized_batch")
    del square_batch
    output = darknet_model.forward(resized_batch)
    del resized_batch
    batch_boxes = utils.get_region_boxes(output, detection_confidence_threshold,
                                   darknet_model.num_classes, darknet_model.anchors, darknet_model.num_anchors)
    for boxes in batch_boxes:
      boxes = utils.nms(boxes, nms_threshold)
      for box in boxes:
        if box[6] == 0:
          human_count += 1
      object_count += len(boxes)
    del batch_boxes
  human_detections_per_frame = human_count / num_frames
  object_detections_per_frame = object_count / num_frames
  results = open('videos_results.txt', 'w+')
  results.write(f'total number of frames: {num_frames}\n')
  results.write(f'mean_human_detections_per_frame: {human_detections_per_frame}\n')
  results.write(f'mean_object_detections_per_frame: {object_detections_per_frame}\n')
  results.close()


if __name__ == "__main__":
  main()

