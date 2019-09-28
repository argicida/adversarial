#!/bin/sh
#
# Downloads model weights (as they're too big to fit in the git repo)

curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolov2.weights
curl https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth -o implementations/ssd_pytorch/weights
