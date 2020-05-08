"""
Example based on Visual iSAM2 example supplied with GTSAM, instead using a custom
wrapper
"""
# pylint: disable=invalid-name, E1101

from __future__ import print_function

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam.examples import SFMdata

from GTSAM_helper import iSAM2Wrapper
import cv2

from vslam_helper import *


if __name__ == '__main__':
    plt.ion()

    K = np.array([[50.0,  0,    50.0],
                      [ 0.0 , 50.0, 50.0],
                      [ 0.0 ,  0.0,  1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    Ksim = gtsam.Cal3_S2(iSAM2Wrapper.CameraMatrix_to_Cal3_S2(K))

    # Create a sample world point, this case it is 10,10,10
    points = SFMdata.createPoints()
    point_3d = points[0]

    # Create 2 camera poses to measure the 3d point from
    poses = SFMdata.createPoses(Ksim)
    pose1 = poses[0]
    pose2 = poses[1]
    
    # Create GTSAM camera objects with poses
    camera1 = gtsam.PinholeCameraCal3_S2(pose1, Ksim)
    camera2 = gtsam.PinholeCameraCal3_S2(pose2, Ksim)
    
    # Project the 3D point into the 2 Camera poses
    kp1 = camera1.project(point_3d)
    kp2 = camera2.project(point_3d)
    
    # Create Transforms of world frame with respect to camera frames
    Pose_1Tw = T_inv(poses[0].matrix())
    Pose_2Tw = T_inv(poses[1].matrix())
    
    # Undistort and nomalize camera measurements
    kp1_ud = cv2.undistortPoints(np.expand_dims(np.expand_dims(kp1.vector(),0),1),K,D)[:,0,:]
    kp2_ud = cv2.undistortPoints(np.expand_dims(np.expand_dims(kp2.vector(),0),1),K,D)[:,0,:]
    
    # make kp 3D arrays
    kp1_ud_3d = np.expand_dims(kp1_ud,1)
    kp2_ud_3d = np.expand_dims(kp2_ud,1)
    
    # create projection matrices frome Transforms
    Proj_1Tw = Pose_1Tw[:3]
    Proj_2Tw = Pose_2Tw[:3]
    
    # Triangulate to calculate the 3D point from the measurements
    pts_3d_hom = cv2.triangulatePoints(Proj_1Tw, Proj_2Tw, kp1_ud_3d, kp2_ud_3d).T
    pts_3d_hom_norm = pts_3d_hom /  pts_3d_hom[:,-1][:,None]
            