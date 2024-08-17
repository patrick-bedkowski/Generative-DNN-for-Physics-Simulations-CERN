# Generative DNN for Physics Simulations CERN

## Table of contents

1. [Setting Developing Environment](#setting-developing-environment)
2. [Producing data for tests](#producing-data-for-tests)
    1. [Original files](#original-files)
    2. [Data filtering](#data-filtering)

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

Original files were generated by the [GEANT4](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13048).
Instructions on how to generate data are available in [here](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideSimulation).<br />
The below files explain the order of steps that need to be performed to run the experiments. 

### Data filtering

The notebook <a href="notebooks/data_filtering.ipynb">data_filtering.ipynb</a> contains the initial preprocessing
and filtering needed for training. It allows you to:
- calculates the photon sum values for images from Proton and Neutron ZDC devices.
- filter the data according to the photon sum values using function `filter_photon_sum()`. <br />This allows you to create datasets to replicate the results in the thesis.
- preprocess the data for the joint model, referred to as padded dataset. <br /> This step adds padding to both images from Proton and Neutron ZDC and concatenates them create image with 2 channels.
- plot the distribution of photon values
- calculate quartile values of photon sum distribution

After the following steps you should have the following files:
- `data_cond_photonsum_proton_X_2312.pkl`
- `data_photonsum_proton_X_2312.pkl`
- `data_proton_neutron_photonsum_proton_18_1970_neutron_18_3249_padding.pkl`
- `data_cond_photonsum_p_18_n_18.pkl`

Where X denotes the minimal value for the photon sum value.

### Calculating diversity among the samples from SDI-GAN implementation

The notebook <a href="notebooks/calculating_diversity_for_data.ipynb">calculating_diversity_for_data.ipynb</a> contains the preprocessing of dataset to calculate diversity of samples explained in Section 8. of the thesis. <br />
You need to use files generated by above script. This is appropriate for both images coming from Proton ZDC device and Padded version of dataset. <br />
After completing the steps you should have the following files:
- `data_cond_stddev_photonsum_p_X.pkl`. Where X denotes the minimal value for the photon sum value in proton data
- `data_cond_stddev_photonsum_p_X_n_X.pkl`. Where X denotes the minimal value for the photon sum value in both proton and neutron data.

### Calculating data for auxiliary regressor
The notebook <a href="notebooks/auxilary regressor/calculate_max_coordinates.ipynb">calculate_max_coordinates.ipynb</a> contains calclation of min max coordinates in images for both Proton and padded dataset.

After completing the steps you should have the following files:
- `data_coord_proton_photonsum_proton_1_2312.pkl`. Where X denotes the minimal value for the photon sum value in proton data
- `data_coord_proton_neutron_photonsum_X.pkl`. Where X denotes the minimal value for the photon sum value in both proton and neutron data.