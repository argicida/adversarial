import os
import numpy as np

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class VideoPipe(Pipeline):
  def __init__(self, batch_size, sequence_length, initial_prefetch_size, nchw, num_threads, device_id, data, shuffle):
    super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)
    self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,
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
  def __init__(self, sequence_length, video_directory, num_threads=4, nchw=True, shuffle=False, device_id=0):
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


def main():
  video_directory = "test_videos/"  
  loader_iterable = TorchVideoFramesLoaderIterable(8, video_directory)
  print(len(loader_iterable))


if __name__ == "__main__":
  main()

