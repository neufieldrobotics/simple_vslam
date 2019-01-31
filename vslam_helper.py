#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 08:36:51 2019
Simple VSLAM Helper File
@author: vik748
"""
import numpy as np
import cv2

def undistortKeyPoints(kps, K, D):
    '''
    This function extracts coordinates from keypoint object,
    undistorts them using K and D and returns undistorted coordinates"
    '''
    #kp_pts = np.array([o.pt for o in kps])
    #kp_pts_cont = np.ascontiguousarray(kp_pts[:,:2]).reshape((kp_pts.shape[0],1,2))
    # this version returns normalized points with F=1 and centered at 0,0
    # cv2.undistortPoints(kp_pts_cont, K, D,  noArray(), K) would return unnormalized output
    return	cv2.undistortPoints(np.expand_dims(kps, axis=1), 
                               cameraMatrix=K, distCoeffs=D)[:,0,:]

def displayMatches(img_left,kp1,img_right,kp2, matches, mask):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    bool_mask = mask.astype(bool)
    img_valid = cv2.drawMatches(img_left,kp1,img_right,kp2,matches, None, 
                                matchColor=(0, 255, 0), 
                                matchesMask=bool_mask.ravel().tolist(), flags=2)
    #img_all = cv2.drawMatches(img_left,kp1,img_right,kp2,matches, img_valid, 
    #                          matchColor=(255, 0, 0), 
    #                          matchesMask=np.invert(bool_mask).ravel().tolist(), 
    #                          flags=1)
    return img_valid

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def plot_pose3_on_axes(axes, gRp, origin, axis_length=0.1):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # get rotation and translation (center)
    #gRp = pose.rotation().matrix()  # rotation from pose to global
    #t = pose.translation()
    #origin = np.array([t.x(), t.y(), t.z()])

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin, x_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin, y_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin, z_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')