# Dependencies
We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed,
to do this see: https://pytorch.org/

To easily set up the dependencies using Python 3 virtual environment, run
```
chmod +x ./setup_venv.sh
./setup_venv.sh
```

Also, since in our git the weights are ignored (too large and not reliably transmitted),
you need to download them manually.

For yolov2, download the weights by executing
```
curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolov2.weights
```

# example training session
```
python3 train_test_patch_one_gpu.py --verbose=True --train_yolov2=1
```

# Tensorboard
To view the training progress remotely through a tensorboard session, first execute
```
chmod +x start_tensorboard_screen_session.sh 
```
, which only needs to be done once on a server.
Then start a persistent session running tensorboard by executing
```
./start_tensorboard_screen_session.sh
```
, which will run the tensorboard server as long as the screen session persists, meaning that you can close the terminal and it will still be accessible.
To access the tensorboard server remotely, on the client machine, execute
```
ssh -N -f -L localhost:8080:<servername>:<tensorboard_port> <sshusername>@<servername>
```
If you don't know the port tensorboard is running on, check it by logging into the server and run `screen -r tensorboard` to log into the screen session, and check for the output from tensorboard. Detach from the screen session without killing it by clicking CTRL+A+D at the same time.

If everything works, you should be able to access the tensorboard by visiting localhost:8080 on your client machine.
