# Tutorials

## Feature Detection and Description
Detect features that are consistently detectable across a number of images. 
Descriptors to uniquely identify features. Choice of feature detector and descriptor 
plays a crucial role in visually degraded images.

In visually degraded images, there's a possibility that detected features get
clumped together in a few spots. This leads to inaccurate homography/ essential matrix 
calculations. Solution is to use tiling or non-max suppression during feature detection

#### Tiling
Divide the image into tiles so the features are distributed throughout the image. 
Refer to feature matching code linked below.

#### Non-Maximum Suppression
Limit the number features that are too close to each other, 
effectively spreading out detected features throughout the image.

[Info on feature detection, matching and adaptive non-maximum suppression](http://www.cs.cornell.edu/courses/cs6670/2011sp/projects/p1/webpages/24/Project%201_Peng%20Chen.htm)

[ANMS code](https://github.com/BAILOOL/ANMS-Codes)

## Feature Matching

Feature matching could be done with a brute force matcher. To filter good matches, 
Lowe's ratio test is performed. Additionally, reverse match is found between image 2 
and image 1 and only if a match appears in both results, it is considered for the 
subsequent steps.

[OpenCV feature matching tutorial](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)

[ORB based feature matching code example](../temp_test_scripts/orb_based_matching.ipynb)

[Zernike feature matching code example](../temp_test_scripts/zernike_matching.ipynb)

## Essential Matrix Calculation
A matrix that relates corresponding points in two images. Five point algorithm to 
calculate Essential matrix. Decomposition of essential matrix to obtain rotation 
and translation

Relative camera rotation and translation can be recovered from the calculated 
essential matrix and the corresponding points. Cheirality check ensures the 
projected points have positive depth

[Essential matrix wiki](https://en.wikipedia.org/wiki/Essential_matrix)

[Recover pose code example](../temp_test_scripts/recover_pose_test.py)

OpenCV function findEssentialMat() calculates the essential matrix and the inliers
recoverPose() calculates relative camera rotation and translation using cheirality check

## Initialization
Relative pose of the first pair of images can be found using the above method - 
Finding essential matrix from feature match and recovering pose. The 
camera pose for the first image would be at origin.

In a monocular setup, since it is not possible to recover scale (scale ambiguity), 
the scale between the first two images is set to one.

## Triangulation
With images from two viewpoints with matching feature points and camera poses, the 
3D object point corresponding to feature points can be determined

[Detailed explanation on triangulation](http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf)

[Code example for triangulation](../temp_test_scripts/triangulation_test.py)

## PnP algorithm
Perspective-n-point algorithm, PnP for short, is for getting the camera 
pose from 3D-2D point correspondences. 

[Detailed explanation on obtaining pose using PnP algorithm](https://docs.opencv.org/4.5.2/dc/d2c/tutorial_real_time_pose.html)

[PnP example using chessboard](../temp_test_scripts/pnp_example_using_chessboard_points.py)
## Tracking and Update
After initialization, each new image could be matched with previous images, obtaining 
point correspondences. The points that have already been triangulated would now have 3D-2D 
correspondence. Using PnP, camera pose can be retrieved.

## GTSAM Backend
To arrive at an optimized solution using GTSAM, a non-linear factor graph has to 
be built by adding measurements and measurement noises. An initial estimate has to 
be provided for the camera poses and landmark positions. The non-linear graph can then 
be optimized using one of the backend optimizers available in GTSAM. Our example uses 
iSAM2 and optimizes after every step.

[GTSAM backend example](../temp_test_scripts/VisualISAM2withWrapper_Example.py)

[Factor graphs and GTSAM](https://gtsam.org/tutorials/intro.html)

[GTSAM 2D SLAM example](http://docs.ros.org/en/melodic/api/gtsam/html/Pose2SLAMExample_8py_source.html)
