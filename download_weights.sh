#!/bin/sh
#
# Downloads model weights (as they're too big to fit in the git repo)

curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolov2.weights
