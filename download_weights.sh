#!/bin/sh
#
# Downloads model weights (as they're too big to fit in the git repo)

curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolov2.weights
curl https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth -o implementations/ssd/models
curl https://pjreddie.com/media/files/yolov3.weights -o implementations/yolov3/weights/yolov3.weights
curl https://pjreddie.com/media/files/darknet53.conv.74 -o implementations/yolov3/weights/darknet53.conv.74
