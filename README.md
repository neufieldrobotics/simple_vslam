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
git checkout simple_vslam_comparisons

# OPTIONALLY To checkout a particular 'tag' eg. v0.2.0 run:
git checkout v0.2.0
```

## Some important versions:
  - v0.1.0: Basic GTSAM incormporated
  - v0.2.0: Cleaned up and configured to use with Iceberg datasets, use -c config_file flag for appropriate config file. 
  - v0.2.1: Major bug fixed, use GTSAM corrected poses on current as well as previous pose to make new point triangulation accurate, this version works with the full Stingray dataset

## Get test data
Test datasets, Unzip and save it to a convenient location like `~/data`: 
  - http://rpg.ifi.uzh.ch/docs/teaching/2016/kitti00.zip
  - Stingray dataset: `deepfreeze1 > /data/datasets_for_algorithms/2018_iceberg_datasets/Stingray/Stingray2_080718_800x600.tar`
  - Lars dataset: `deepfreeze1 > /data/datasets_for_algorithms/2018_iceberg_datasets/Lars/Lars2_081018_800x600.zip`
  - Cervino dataset: `deepfreeze1 > /data/datasets_for_algorithms/2018_iceberg_datasets/Cervino/time_lapse_5_cervino_800x600.zip`

## Executing the package
Edit a config file to point it to an image folder eg.
```sh
gedit config/kitti.conf
```
Depending on the system, make sure either osx_image_folder or linux_image_folder points to the appropriate image folder. You can also choose the feature detector and descriptor to be used (independently). Then run the code with:
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

simple_vslam_comparisons