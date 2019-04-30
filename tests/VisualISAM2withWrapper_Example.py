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


if __name__ == '__main__':
    plt.ion()

    # Define the camera calibration parameters
    # K = gtsam.Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

    K = np.array([[50.0,  0,    50.0],
                      [ 0.0 , 50.0, 50.0],
                      [ 0.0 ,  0.0,  1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    Ksim = gtsam.Cal3_S2(iSAM2Wrapper.CameraMatrix_to_Cal3_S2(K))

    # Define the camera observation noise model
#    measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
#        2, 1.0)  # one pixel in u and v

    # Create the set of ground-truth landmarks
    points = SFMdata.createPoints()

    # Create the set of ground-truth poses
    poses = SFMdata.createPoses(Ksim)
    
    factor_graph = iSAM2Wrapper(poses[0].matrix(), pose0_to_pose1_range=17,
                                relinearizeThreshold=0.1, relinearizeSkip=1, K=np.eye(3), 
                                proj_noise_val=1.0/100)
    #factor_graph.set_Camera_matrix(np.eye(3))
    #factor_graph.set_Projection_noise(1.0/100)
    

    # Create a Factor Graph and Values to hold the new data
    #  Loop over the different poses, adding the observations to iSAM incrementally
    for i, pose in enumerate(poses):

        # Add factors for each landmark observation
        for j, point in enumerate(points):
            camera = gtsam.PinholeCameraCal3_S2(pose, Ksim)
            measurement = camera.project(point)
            undist_m = cv2.undistortPoints(np.expand_dims(np.expand_dims(iSAM2Wrapper.Point2arr(measurement),0),1),K,D)[:,0,:]
            #undist_pt = iSAM2Wrapper.arr2Point(undist_m)

            #m_pt = np.array([measurement.x(), measurement.y()])
            factor_graph.add_GenericProjectionFactorCal3_S2_factor(undist_m, i, np.array([j]))

        # Add an initial guess for the current pose
        # Intentionally initialize the variables off from the ground truth
        pose_est = pose.compose(gtsam.Pose3(gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), 
                                            gtsam.Point3(0.05, -0.10, 0.20)))        
        if i != 0:
            factor_graph.add_PoseEstimate(i, pose_est.matrix() )

        # If this is the first iteration, add a prior on the first pose to set the
        # coordinate frame and a prior on the first landmark to set the scale.
        # Also, as iSAM solves incrementally, we must wait until each is observed
        # at least twice before adding it to iSAM.
        if i == 0:
            # Add initial guesses to all observed landmarks
            # Intentionally initialize the variables off from the ground truth
            for j, point in enumerate(points):
                factor_graph.add_LandmarkEstimate(np.array([j]), np.array([[point.x()-0.25, point.y()+0.20, point.z()+0.15]]))
        else:
            # Update iSAM with the new factors
            # Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
            # If accuracy is desired at the expense of time, update(*) can be called additional
            # times to perform multiple optimizer iterations every step.
            factor_graph.update(3)
            
            current_estimate = factor_graph.get_Estimate()
            factor_graph.isam2.saveGraph("ex_iter_"+str(i)+".dot")
            '''
            print("****************************************************")
            print("Frame", i, ":")
            for j in range(i + 1):
                print('X',j, ":", current_estimate.atPose3(iSAM2Wrapper.get_key('x',j)))

            for j in range(len(points)):
                print('L',j, ":", current_estimate.atPoint3(iSAM2Wrapper.get_key('l',j)))
            '''
            plt.pause(1)

            factor_graph.plot_estimate(1)
            #plt.figure(1).get_axes().set_aspect('equal')

    plt.ioff()
    plt.show()

