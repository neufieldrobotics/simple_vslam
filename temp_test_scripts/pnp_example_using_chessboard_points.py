#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:00:20 2019

@author: vik748
"""
import os,sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# This allows adding correct path whether run from file, spyder or notebook
try:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
except NameError as e:
    print(e)
    this_file_dir = globals()['_dh'][0]
    
sys.path.insert(0, os.path.join(this_file_dir, os.pardir,'helper_functions'))
from vslam_helper import *
np.set_printoptions(suppress=True)

# Inputs, images and camera info
    
data_path = os.path.join(this_file_dir,os.pardir,'data')
img_folder = 'gopro_chess_board_800x600'

img1 = cv2.imread(os.path.join(data_path, img_folder,'GOPR1550.jpg'),cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(data_path, img_folder,'GOPR1551.jpg'),cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(os.path.join(data_path, img_folder,'GOPR1552.jpg'),cv2.IMREAD_GRAYSCALE)

print("img1_path:",os.path.join(data_path, img_folder,'GOPR1550.jpg'))
K = np.array([[700.551256,   0.     ,  403.995336],
              [  0.      , 695.41896,  289.95235 ],
              [  0.      ,   0.     ,    1.0     ]])
 
D = np.array([[-0.28571128, 0.16130412, 0.00005361, -0.00014855, -0.07717838]])

CHESSBOARD_W = 16
CHESSBOARD_H = 9
CHESSBOARD_SIZE = 0.08075

print(K,D)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_W*CHESSBOARD_H,3), np.float32)
#objp[:,:2] = np.mgrid[0:CHESSBOARD_W,0:CHESSBOARD_H].T.reshape(-1,2)
objp[:,[0,2]] = np.mgrid[0:CHESSBOARD_W,0:CHESSBOARD_H].T.reshape(-1,2)
objp = objp * CHESSBOARD_SIZE

#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

imgs = [img1,img2,img3]

fig, ax = initialize_3d_plot(number=1, title='Ideal chess board points vs estimated camera positions')

graph = plot_3d_points(ax, objp, linestyle="", marker="o")

Cs = []
Ts = []

for img_cur in imgs:
    #gray = cv2.cvtColor(img_cur,cv2.COLOR_BGR2GRAY)
    gray = img_cur
    ret, corners = cv2.findChessboardCorners(gray, (16,9),None)
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    Cs.append(corners2)
    success, T, inliers = T_from_PNP(objp, corners2, K, D)
    Ts.append(T)
    plot_pose3_on_axes(ax,T, axis_length=.25)

set_axes_equal(ax)  
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")

T_0_1 = Ts[0]
T_0_2 = Ts[1]
T_0_3 = Ts[2]

T_1_1 = np.eye(4) # make camera 1 at origin

 # calculate camera 2 and 3 in camera 1's frame
T_1_0 = T_inv(T_0_1)
T_1_2 = T_1_0 @ T_0_2
T_1_3 = T_1_0 @ T_0_3

T_2_0 = T_inv(T_0_2)
T_2_3 = T_2_0 @ T_0_3

fig, ax = initialize_3d_plot(number=2, title='Chess board points trignulated using camera positions \n Camera 1 at origin', view=(-70, -90))

corners1_ud = cv2.undistortPoints(Cs[0],K,D)
corners2_ud = cv2.undistortPoints(Cs[1],K,D)
corners3_ud = cv2.undistortPoints(Cs[2],K,D)

# Plot camera 1 , camera 2 and trigulated chess board corners between 1 and 2 in green
corners_12,_ = triangulate(T_1_1, T_1_2, corners1_ud, corners2_ud )
graph = plot_3d_points(ax, corners_12, linestyle="", marker=".",color='g')
plot_pose3_on_axes(ax,T_1_1, axis_length=1.0)
plot_pose3_on_axes(ax,T_1_2,axis_length=1.0)

# Plot camera 3 and trigulated chess board corners between 1 and 3 in red
corners_13,_ = triangulate(T_1_1, T_1_3, corners1_ud, corners3_ud )
graph = plot_3d_points(ax, corners_13, linestyle="", marker=".",color='r')

# Plot rigulated chess board corners between 2 and 3 in cyan
plot_pose3_on_axes(ax,T_1_3,axis_length=1.0)
corners_23,_ = triangulate(T_1_2, T_1_3, corners2_ud, corners3_ud )
graph = plot_3d_points(ax, corners_23, linestyle="", marker=".",color='c')

set_axes_equal(ax)
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")
