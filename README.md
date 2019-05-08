# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks(DCGAN)
-------------------------------------------------
## Introduction
 * The code is adapted from the PyTorch documentation examples:<br>
<https://github.com/pytorch/examples/tree/master/dcgan> <br>
 * The file download_lsun.py comes from a nice repository for downloading LSUN dataset:
<https://github.com/fyu/lsun> <br>

 * I have added massive comments for the code. Hope it beneficial for understanding GANs/DCGANs, especially for a beginner.

## Environment & Requirements
* CentOS Linux release 7.2.1511 (Core)<br>
* python 3.6.5<br>
* torch  1.0.0<br>
* torchvision<br>
* argparse<br>
* os<br>
* random<br>
* subprocess<br>
* urllib

## Usage
### Train DCGAN with MNIST:<br>
    ```python
    python3 main.py --dataset mnist --cuda
    ```
   <br>
Two folders will be created, i.e., *data* & *results*. The *data* folder stores dataset. <br>
The *results* folder stores the generated images and the trained models.<br> 
You can also use cifar10, lsun, imagenet, randomly generated fake data, etc.
### Download lsun dataset:<br>
    ```python
    python3 download_lsun.py
    ```
   <br>
Download the whole data set and save it to ./data.<br>
    ```python
    python3 download_lsun.py --category bedroom 
    ```
   <br>
Download data for bedroom.<br> By replacing the option of *--category*, you can download data of each category in LSUN as well.

## NOTE
 * The DCGAN architecture is a relatively primary version. Now there exists some new modifications.<br> 
 * The batch_size, size of feature maps of both G and D are all set to 64, different from that in the paper (128).<br>With above hyperparameters set to 128, the model confronts gradient vanishing. Hope someone help me with it.
 
## References 
1. PyTorch documentation
2. <https://github.com/pytorch/examples/tree/master/dcgan> <br>
3. <https://github.com/fyu/lsun> <br>
