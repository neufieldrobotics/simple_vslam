#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:00:20 2019

@author: vik748
"""
import sys
import cv2
import numpy as np
from numpy.linalg import inv
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vslam_helper import *

# Load previously saved data
# Inputs, images and camera info
if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/chess_board2/GOPR1496.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board2/GOPR1497.JPG',1)  
img3 = cv2.imread(path+'data/chess_board2/GOPR1498.JPG',1)
'''
fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])
'''    
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


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def pose_inv(R_in, t_in):
    t_out = -np.matmul((R_in).T,t_in)
    R_out = R_in.T
    return R_out,t_out

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))

def compose_T(R,t):
    return np.vstack((np.hstack((R,t)),np.array([0, 0, 0, 1])))

def decompose_T(T_in):
    return T_in[:3,:3], T_in[:3,[-1]]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_W*CHESSBOARD_H,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_W,0:CHESSBOARD_H].T.reshape(-1,2)
objp = objp * CHESSBOARD_SIZE

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

imgs = [img1,img2,img3]
#img_cur = img1

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')

Rs = []
ts = []
Cs = []

for img_cur in imgs:
    gray = cv2.cvtColor(img_cur,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (16,9),None)
    
    
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    Cs.append(corners2)
    
    # Find the rotation and translation vectors.
    success, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, K, D)
    
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, D)
    
    imgout = draw(img_cur,corners2,imgpts)
    #plt.imshow(imgout)
    #plt.show()
    
    graph, = ax.plot(objp[:,0], objp[:,1], objp[:,2], linestyle="", marker="o")
    Rinv, J	=	cv2.Rodrigues(rvecs)
    
    #CP = np.matmul(-Rinv.T,tvecs)
    R,t = pose_inv(Rinv,tvecs)
    Rs.append(R)
    ts.append(t)
    
    plot_pose3_on_axes(ax,R,t.T, axis_length=.25)
    #plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=1.0)

set_axes_equal(ax)  
#plt.show()
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")

fig = plt.figure(2)
T_0_1 = compose_T(Rs[0],ts[0])
T_0_2 = compose_T(Rs[1],ts[1])
T_0_3 = compose_T(Rs[2],ts[2])

T_1_0 = T_inv(T_0_1)
T_1_2 = np.matmul(T_1_0 , T_0_2)
T_1_3 = np.matmul(T_1_0 , T_0_3)

Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
print ("Pose_1: ", Pose_1)
R_1_2 = T_1_2[:3,:3]
t_1_2 = T_1_2[:3,[-1]]
Pose_2 = np.hstack((R_1_2.T,-t_1_2))
print ("Pose_2: ", Pose_2)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
plt.get_current_fig_manager().window.setGeometry(992, 430, 928, 1028)
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')


corners1_ud = cv2.undistortPoints(Cs[0],K,D)
corners2_ud = cv2.undistortPoints(Cs[1],K,D)
corners3_ud = cv2.undistortPoints(Cs[2],K,D)

corners_12_hom = cv2.triangulatePoints(Pose_1, Pose_2, corners1_ud, corners2_ud).T
corners_12_hom_norm = corners_12_hom /  corners_12_hom[:,-1][:,None]
corners_12 = corners_12_hom_norm[:, :3]

graph = plot_3d_points(ax, corners_12, linestyle="", marker=".",color='g')
plot_pose3_on_axes(ax,Pose_1[:,:3], Pose_1[:,[-1]].T,axis_length=1.0)
plot_pose3_on_axes(ax,Pose_2[:,:3].T, -Pose_2[:,[-1]].T,axis_length=1.0)

R_1_3 = T_1_3[:3,:3]
t_1_3 = T_1_3[:3,[-1]]
Pose_3 = np.hstack((R_1_3.T,-t_1_3))
print ("Pose_3: ", Pose_3)


corners_23_hom = cv2.triangulatePoints(Pose_2, Pose_3, corners2_ud, corners3_ud).T
corners_23_hom_norm = corners_23_hom /  corners_23_hom[:,-1][:,None]
corners_23 = corners_23_hom_norm[:, :3]
graph = plot_3d_points(ax, corners_23, linestyle="", marker=".",color='r')
plot_pose3_on_axes(ax,Pose_3[:,:3].T, -Pose_3[:,[-1]].T,axis_length=1.0)


set_axes_equal(ax)
plt.show()

