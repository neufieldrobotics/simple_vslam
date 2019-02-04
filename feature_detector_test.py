#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import cv2
import sys

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,2) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1],
                ...,
                [xn,yn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keept or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter


def tiled_features(kp, img, tiley, tilex):
    feat_per_cell = int(len(kp)/(tilex*tiley))
    HEIGHT, WIDTH = img.shape
    assert WIDTH%tiley == 0, "Width is not a multiple of tilex"
    assert HEIGHT%tilex == 0, "Height is not a multiple of tiley"
    w_width = int(WIDTH/tiley)
    w_height = int(HEIGHT/tilex)
        
    xx = np.linspace(0,HEIGHT-w_height,tilex,dtype='int')
    yy = np.linspace(0,WIDTH-w_width,tiley,dtype='int')
        
    kps = np.array([])
    pts = np.array([keypoint.pt for keypoint in kp])
    kp = np.array(kp)
    
    for ix in xx:
        for iy in yy:
            inbox_mask = bounding_box(pts, ix,ix+w_height,iy,iy+w_height)
            inbox = kp[inbox_mask]
            inbox_sorted = sorted(inbox, key = lambda x:x.response, reverse = True)
            inbox_sorted_out = inbox_sorted[:feat_per_cell]
            print(np.shape(inbox), ' and ', np.shape(inbox_sorted_out))
            kps = np.append(kps,inbox_sorted_out)
            #print(kps)
    return kps.tolist()


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 15, color)
    return vis


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img = cv2.imread(path+'data/chess_board/GOPR1488.JPG',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img = cv2.imread(path+'data/test_set/GOPR1429.JPG',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

mask_pts_1488 = np.array([[1180, 960], [2740, 1000], [2700, 2040], [1180, 1980]])
mask_pts_1489 = np.array([[1180, 1225], [2550, 1475], [2400, 2340], [1100, 2100]])
mask_pts_1490 = np.array([[1550, 1250], [2870, 1030], [2900, 1890], [1680, 2200]])
mask_pts_1491 = np.array([[760, 880], [2100, 1090], [2180, 1980], [780, 2100]])

    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create a mask image filled with zeros, the size of original image
mask = np.zeros(img.shape[:2], dtype=np.uint8)

mask = cv2.fillConvexPoly(mask, mask_pts_1488, color=[255, 255, 255])
mask = 255-mask

tiley = 16
tilex = 12
total_feat = 25000

detector = cv2.ORB_create(nfeatures=int(25000), edgeThreshold=32, patchSize=32, nlevels=16, 
                     fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=8)

# find the keypoints and descriptors with ORB
kp = detector.detect(img,mask)
#kpo = tiled_features(kp, gray, tiley, tilex)



img = draw_keypoints(img,kp)           
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', (800,600))
cv2.imshow('image', img)
cv2.waitKey()