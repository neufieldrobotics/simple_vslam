# simple_vslam metashape_pts

This branch metashape_pts is intended as a test for stability and accuracy of the 
gtsam / isam2 implementation. It uses feature pts exported from an aligned metashape 
project.

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
git checkout metashape_pts
```

## Get test data
Test datasets, Unzip and save it to a convenient location like `~/data`: 
  - Lars1 dataset: 
  - Stingrat2 dataset:
  - Cervino dataset:

## Create pkl files from metashape
In order to create the required pkl file from Metashape use the function 'export_points_and_cams_to_pkl' from https://gitlab.com/neufieldrobotics/metashape_scripts

## Executing the package
Edit a config file to point it to an image folder eg.
```sh
gedit config/go_pro_Stingray2_metashape.conf
```
Depending on the system, make sure either osx_image_folder or linux_image_folder points to the appropriate image folder. Then run the code with:
```sh
./vslam_metashape.py -c config/go_pro_Stingray2_metashape.conf
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

