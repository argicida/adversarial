#!/bin/bash
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install torch
pip install pillow
pip install tqdm
pip install matplotlib
pip install tensorboardx
pip install tensorboard
pip install opencv-python
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
deactivate
