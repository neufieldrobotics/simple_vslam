#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:47:19 2019

@author: vik748
"""
import numpy as np
import cv2
import glob
import sys
from matplotlib import pyplot as plt
np.set_printoptions(precision=8,suppress=False)


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
CHESSBOARD_W = 16
CHESSBOARD_H = 9
CHESSBOARD_SIZE = 0.08075 #0.0355 #0.08075

objp = np.zeros((CHESSBOARD_W*CHESSBOARD_H,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_W,0:CHESSBOARD_H].T.reshape(-1,2)
objp = objp * CHESSBOARD_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(path+'data/goprocalib_80.75mm_target_set_2/*.JPG')
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', (800,600))

fig = plt.figure(1)


for fname in images:
    print ("File - ", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_W,CHESSBOARD_H),None)

    # If found, add object points, image points (after refining them)
    print("ret: ",ret," Pts: ", corners.shape)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (CHESSBOARD_W,CHESSBOARD_H), corners2,ret)
        
        plt.imshow(img)
        #plt.ion()
        #plt.show()
        plt.draw()
        plt.pause(0.5)
        #input("Press [enter] to continue.")


        # Draw and display the corners
        #
        #cv2.imshow('image',img)
        #cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Img pts shape: ",len(imgpoints))
print("Camera matrix: ", mtx)
print("Dist: ",dist)
print("ret: " , ret)
plt.close(fig='all')