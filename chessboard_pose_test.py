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
    img1 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1429.JPG',1)          # queryImage
    img2 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1430.JPG',1)  
    img3 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1431.JPG',1)  
else:    
    img1 = cv2.imread('/home/vik748/data/chess_board/GOPR1460.JPG',1)          # queryImage
    img2 = cv2.imread('/home/vik748/data/chess_board/GOPR1461.JPG',1)  
    img3 = cv2.imread('/home/vik748/data/chess_board/GOPR1462.JPG',1)

fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])

D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);

print(K,D)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*16,3), np.float32)
objp[:,:2] = np.mgrid[0:16,0:9].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img_cur = img3

gray = cv2.cvtColor(img_cur,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (16,9),None)


corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

# Find the rotation and translation vectors.
success, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, K, D)

# project 3D points to image plane
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, D)

imgout = draw(img_cur,corners2,imgpts)
#plt.imshow(imgout)
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
graph, = ax.plot(objp[:,0], objp[:,1], objp[:,2], linestyle="", marker="o")
R, J	=	cv2.Rodrigues(rvecs)

CP = np.matmul(-R.T,tvecs)

plot_pose3_on_axes(ax,R.T,CP.T, axis_length=1.0)
#plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=1.0)

set_axes_equal(ax)  
plt.show()
#k = cv2.waitKey(0) & 0xff

#cv2.destroyAllWindows()