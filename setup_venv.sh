#!/bin/bash
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install torch
pip install pillow
pip install torchvision
pip install tqdm
pip install matplotlib
pip install tensorboardx
pip install tensorboard
deactivate