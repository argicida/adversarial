# Dependencies
We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed,
to do this see: https://pytorch.org/

To easily set up the dependencies using Python 3 virtual environment, run
```
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