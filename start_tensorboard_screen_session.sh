#!/bin/bash
screen -d -m -S tensorboard bash -c 'source ./venv/bin/activate;tensorboard --logdir=./runs;'
