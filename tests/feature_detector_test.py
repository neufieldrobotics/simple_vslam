#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

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
    print(xx)
    yy = np.linspace(0,WIDTH-w_width,tiley,dtype='int')
    print(yy)
        
    kps = np.array([])
    pts = np.array([keypoint.pt for keypoint in kp])
    kp = np.array(kp)
    
    for ix in xx:
        for iy in yy:
            inbox_mask = bounding_box(pts, iy,iy+w_height, ix,ix+w_height)
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
        cv2.circle(vis, (int(x), int(y)), 20, color, thickness=3)
    return vis

def draw_markers(vis, keypoints, color = (0, 0, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.drawMarker(vis, (int(x), int(y)), color,  markerSize=30, markerType = cv2.MARKER_CROSS, thickness=2)
    return vis

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img = cv2.imread(path+'data/kitti/00/image_0/000000.png',0) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img = cv2.imread(path+'data/test_set/GOPR1429.JPG',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

mask_pts_1488 = np.array([[1180, 960], [2740, 1000], [2700, 2040], [1180, 1980]])
mask_pts_1489 = np.array([[1180, 1225], [2550, 1475], [2400, 2340], [1100, 2100]])
mask_pts_1490 = np.array([[1550, 1250], [2870, 1030], [2900, 1890], [1680, 2200]])
mask_pts_1491 = np.array([[760, 880], [2100, 1090], [2180, 1980], [780, 2100]])
mask_pts_1496 = np.array([[400, 650], [1300, 680], [1280, 1300], [390, 1260]])
mask_pts_1497 = np.array([[760, 1055], [1600, 1080], [1560, 1670], [755, 1630]])
mask_pts_1498 = np.array([[1030, 740], [1820, 840], [1785, 1085], [1675, 1400], [975, 1285]])
    
gray = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create a mask image filled with zeros, the size of original image
mask = np.ones(img.shape[:2], dtype=np.uint8)

mask = cv2.fillConvexPoly(mask, mask_pts_1498, color=[0, 0, 0])
#mask = 255-mask

tiley = 17
tilex = 4
total_feat = 25000

# defailts: int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, 
# int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, 
# int patchSize=31, int fastThreshold=20)

detector1 = cv2.ORB_create(nfeatures=1000, edgeThreshold=125, patchSize=125, nlevels=8, 
                     fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

detector2 = cv2.ORB_create(nfeatures=1000, edgeThreshold=65, patchSize=65, nlevels=4, 
                     fastThreshold=20, scaleFactor=4.0, WTA_K=4,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

# find the keypoints and descriptors with ORB
#kp1 = detector1.detect(img,mask)
kp2 = detector2.detect(img,mask)
kp2p = tiled_features(kp2, gray, tiley, tilex)

img1 = draw_keypoints(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),kp2)
img2 = draw_markers(img1,kp2p)
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', (800,600))
#cv2.imshow('image', img)
#cv2.waitKey()

fig2 = plt.figure(2)
plt.axis("off")
plt.imshow(img2)
plt.show()
