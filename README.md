# ExpertSim: Fast Particle Detector Simulation Using Mixture-of-Generative-Experts

## Table of contents

1. [Setting Developing Environment](#setting-developing-environment)
2. [Producing data for tests](#producing-data-for-tests)
    1. [Original files](#original-files)

## Setting Developing Environment

In order to run all experiments and analysis jupyter notebooks, it is necessary to install the following tools: <br />Download version for your operating system.
1. [Python 3.9.16 download](https://www.python.org/downloads/release/python-3916/)
2. Install CUDA: <br />
    Note, for your system to actually use the GPU, it nust have a [Compute Capibility](https://developer.nvidia.com/cuda-gpus) >= to 3.0<br />
    Install CUDA 11.7 for your OS
   1. [CUDA Toolkit 11.7 Downloads](https://developer.nvidia.com/cuda-11-7-0-download-archive)<br />
   * Windows:<br /> double-click the executable and follow setup instructions<br />
   * Linux:<br /> follow the instructions [here](http://askubuntu.com/a/799185)<br />
   2. [Download cuDNN v8.9.6 (November 1st, 2023), for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)
3. Install python pip modules from `requirements.txt` using command:
```pip install -r requirements.txt```

After the above setup it should be possible to run the scripts.


## Producing data for tests

In order to run all experiments, it is necessary to build datasets from original files.

Necessary data to run the experiments are the following:
- dataset with 9 conditional variables describing the: Mass, Energy, Charge, 3 vectors for momenta and 3 vectors for coordinates
- dataset with images originating from Proton ZDC device
- dataset with images originating from Neutron ZDC device

### Original files

To get the original files needed for training and inference get it from here [Google Drive](). 

### Training instructions

1. Open ```expertsim/train/hooks.py``` file and fill variables `ENTITY` and `PROJECT` with your [Weights & Biases](https://wandb.ai/) account name and project name.
2. Modify the config files ```expertsim/config/default.yaml```
3. Run the training from main directory using command: ```python -u cli.py --config expertsim/config/default.yaml```
