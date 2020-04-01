# simple_vslam

simple_vslam is a python based implementation of visual slam using opencv.
## Prerequisites
The following python packages are required to run this code:
  - opencv
  - matplotlib
  - numpy
  - scipy
  - yaml
  - colorama

The code has been tested with the following conda environment:
```sh
conda create -n simple_vslam_env python=3.5 opencv=3.3.1 matplotlib numpy scipy pyyaml colorama progressbar2
conda activate simple_vslam_env
```
## Installation
```sh
git clone --recurse-submodules https://gitlab.com/neu-mit-lfi/simple_vslam.git
cd simple_vslam
git checkout zernike_gtsam

# To checkout a particular 'tag' eg. v0.2.0 run:
git checkout v0.2.0
```
### GTSAM Installation
Main repo https://bitbucket.org/gtborg/gtsam/overview

Note: The GTSAM Overview lists Intel Math Kernel Library as optional, but **DO NOT INSTALL IT**. If Intel Math Kernel Library is installed, GTSAM compilation expects to use it. However, the standard Python doesn't use Math Kernel Library and will cause issues. So if you want to use the Intel Math Kernel Library, make sure it works with your python installation.

```sh
# On termial go into the conda environment by running
conda activate simple_vslam_env

# Check which Python the system is using by running:
which python
# The output should be the one from the conda environment on mac /Users/<username>/anaconda3/envs/simple_vslam_env/bin/python

# Also check the current PYTHONPATH by running:
echo $PYTHONPATH
# On my mac it was empty

# Install pre-reqs, you might have most of these already
# for linux
sudo apt install python-pip cmake libboost-all-dev

# The original gtsam code can be cloned with git clone https://github.com/borglab/gtsam.git ~/apps/gtsam
# However, the system was last tested with commit e7632608e59e6378b2103366a584f17472a1a71e
git clone https://github.com/borglab/gtsam.git
cd gtsam
# optional, checkout a specific commit with git checkout <commit_hash>
# Install gtsam python requirements with
pip install -r cython/requirements.txt 

# Edit the CMakeLists.txt file to the flags GTSAM_BUILD_PYTHON, GTSAM_INSTALL_MATLAB_TOOLBOX and GTSAM_INSTALL_CYTHON_TOOLBOX turn off GTSAM_ALLOW_DEPRECATED_SINCE_V4, 
# the respective lines should look as follows:
  
option(GTSAM_ALLOW_DEPRECATED_SINCE_V4   "Allow use of methods/functions deprecated in GTSAM 4" OFF)
option(GTSAM_INSTALL_MATLAB_TOOLBOX      "Enable/Disable installation of matlab toolbox"  ON)
option(GTSAM_INSTALL_CYTHON_TOOLBOX      "Enable/Disable installation of Cython toolbox"  ON)

# These instructions are for installing within a conda environment, for system wide installation, this would have to be adjusted accordingly:  
mkdir build
cd build
cmake -D GTSAM_PYTHON_VERSION=3.5 -DCMAKE_INSTALL_PREFIX=/Users/<username>/gtsam_bin ..
make check -j4 # (check optional, runs unit tests)
# Takes about 30 mins.
make install # this should install the binaries to /Users/<username>/gtsam_bin

# test installation with:
python /Users/<username>/gtsam_bin/cython/gtsam/examples/ImuFactorExample2.py

# If using conda environments, add the path '/Users/<username>/gtsam_bin/cython' to a .pth file in 
# '/Users/<username>/anaconda/envs/simple_vslam_env/lib/python3.5/site-packages/gtsam.pth'
```

## Some important versions:
  - v0.1.0: Basic GTSAM incormporated
  - v0.2.0: Cleaned up and configured to use with Iceberg datasets, use -c config_file flag for appropriate config file. 

## Get test data
Test datasets, Unzip and save it to a convenient location like `~/data`: 
  - http://rpg.ifi.uzh.ch/docs/teaching/2016/kitti00.zip  
  - Lars dataset: deepfreeze1 > /data/datasets_for_algorithms/2018_iceberg_datasets/Lars2_081018_800x600.zip
  - Cervino dataset: deepfreeze1 > /data/datasets_for_algorithms/2018_iceberg_datasets/time_lapse_5_cervino_800x600.zip

## Executing the package
Edit a config file to point it to an image folder eg.
```sh
gedit config/kitti.conf
```
Depending on the system, make sure either osx_image_folder or linux_image_folder points to the appropriate image folder. Then run the code with:
```sh
./vslam.py -c config/kitti.conf
```

## Algorithm
### Frame Dataflow
<img src="docs/frame_data_flow.png" alt="Simaple Vslam Frame Data Flow" width="800" align="middle">

### iSAM2 algorithm
1. Process X<sub>0</sub>  
   <img src="https://gitlab.com/neu-mit-lfi/simple_vslam/raw/zernike_gtsam/docs/gtsam_workflow_diagrams/gtsam_diagram_x0.svg" alt="GTSAM diagram x0" width="800" align="middle">
    1. Add prior factor for X<sub>0</sub>
    2. Add estimate for X<sub>0</sub>
1. Process X<sub>1</sub>  
   <img src="https://gitlab.com/neu-mit-lfi/simple_vslam/raw/zernike_gtsam/docs/gtsam_workflow_diagrams/gtsam_diagram_x1.svg" alt="GTSAM diagram x1" width="800" align="middle">
    1. Add range factor between X<sub>0</sub> - X<sub>1</sub>  
    3. Add estimate for X<sub>1</sub>
1. Process X<sub>2</sub>  
   <img src="https://gitlab.com/neu-mit-lfi/simple_vslam/raw/zernike_gtsam/docs/gtsam_workflow_diagrams/gtsam_diagram_x2.svg" alt="GTSAM diagram x2" width="800" align="middle">
    1. Add projection factor between X<sub>2</sub> - l<sub>2</sub>, X<sub>1</sub> - l<sub>2</sub> & X<sub>0</sub> - l<sub>2</sub>
    2. Add estimate for X<sub>2</sub> & l<sub>2</sub>

1. Process X<sub>3</sub>  
   <img src="https://gitlab.com/neu-mit-lfi/simple_vslam/raw/zernike_gtsam/docs/gtsam_workflow_diagrams/gtsam_diagram_x3.svg" alt="GTSAM diagram x3" width="800" align="middle">
    1. Add projection factor between X<sub>3</sub> - l<sub>4</sub>, X<sub>2</sub> - l<sub>4</sub> & X<sub>1</sub> - l<sub>4</sub>
    2. Add estimate for X<sub>3</sub> & l<sub>4</sub>

