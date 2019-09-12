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

# Testing Scenarios
### Reproduce Original Research
`patch_config.py` contains configuration of different experiments.
You can design your own experiment by inheriting from the base `BaseConfig`
class or an existing experiment. `ReproducePaperObj` reproduces the patch that
minimizes object score from the paper (With a lower batch size to fit on a
desktop GPU).

You can generate this patch by running:
```
python train_patch.py paper_obj
```
### Ten Times the Original Target Size
This one challenges the hardware by replicating the yolov2 target architecture ten times
You can simulate the situation by running:
```
python train_patch_10x.py paper_obj
```
### Patch Testing
Not sure what it really does yet, but it iterates through tested images and apply patch onto them, and saves its output
to clean_results.json, noise_results.json, and patch_results.json. It also creates a ton of txt and png files in testing/
```
python test_patch.py
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
