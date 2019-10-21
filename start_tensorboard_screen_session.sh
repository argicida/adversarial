#!/bin/bash
screen -d -m -S tensorboard bash -c 'source ./venv/bin/activate;tensorboard --port=4567 --logdir=./runs;'
