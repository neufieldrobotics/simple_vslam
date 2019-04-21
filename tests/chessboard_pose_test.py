#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:00:20 2019

@author: vik748
"""
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vslam_helper import *

# Load previously saved data
# Inputs, images and camera info
if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/chessboard_triangulation/GOPR1531.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chessboard_triangulation/GOPR1532.JPG',1)  
img3 = cv2.imread(path+'data/chessboard_triangulation/GOPR1533.JPG',1)

K = np.array([[3.50275628e+03, 0.00000000e+00, 2.01997668e+03],
              [0.00000000e+00, 3.47709480e+03, 1.44976175e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    #D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);
D = np.array([[-2.85711277e-01,  1.61304120e-01,  5.36070359e-05, -1.48554708e-04,
               -7.71783829e-02]])

CHESSBOARD_W = 16
CHESSBOARD_H = 9
CHESSBOARD_SIZE = 0.08075

print(K,D)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_W*CHESSBOARD_H,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_W,0:CHESSBOARD_H].T.reshape(-1,2)
objp = objp * CHESSBOARD_SIZE

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

imgs = [img1,img2,img3]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
graph = plot_3d_points(ax, objp, linestyle="", marker="o")

Cs = []
Ts = []

for img_cur in imgs:
    gray = cv2.cvtColor(img_cur,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (16,9),None)
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    Cs.append(corners2)
    success, T, inliers = T_from_PNP(objp, corners2, K, D)
    Ts.append(T)
    plot_pose3_on_axes(ax,T, axis_length=.25)

set_axes_equal(ax)  
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")

fig = plt.figure(2)
T_0_1 = Ts[0]
T_0_2 = Ts[1]
T_0_3 = Ts[2]

T_1_0 = T_inv(T_0_1)
T_1_2 = T_1_0 @ T_0_2
T_1_3 = T_1_0 @ T_0_3

T_2_0 = T_inv(T_0_2)
T_2_3 = T_2_0 @ T_0_3

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
plt.get_current_fig_manager().window.setGeometry(992, 430, 928, 1028)
ax.set_aspect('equal')         # important!
title = ax.set_title('Plotting with image 1 at center')

corners1_ud = cv2.undistortPoints(Cs[0],K,D)
corners2_ud = cv2.undistortPoints(Cs[1],K,D)
corners3_ud = cv2.undistortPoints(Cs[2],K,D)

corners_12 = triangulate(T_1_2, T_1_2, corners1_ud, corners2_ud )
graph = plot_3d_points(ax, corners_12, linestyle="", marker=".",color='g')
plot_pose3_on_axes(ax,np.eye(4), axis_length=1.0)
plot_pose3_on_axes(ax,T_1_2,axis_length=1.0)

corners_13 = triangulate(T_1_3, T_1_3, corners1_ud, corners3_ud )
graph = plot_3d_points(ax, corners_13, linestyle="", marker=".",color='r')
plot_pose3_on_axes(ax,T_1_3,axis_length=1.0)

corners_23 = triangulate(T_1_2, T_1_3, corners2_ud, corners3_ud )
graph = plot_3d_points(ax, corners_23, linestyle="", marker=".",color='c')

set_axes_equal(ax)
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")