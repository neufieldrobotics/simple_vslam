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
import os
import glob
from zernike.zernike import MultiHarrisZernike

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

def match_image_names(set1, set2):
    '''Return true if images in set2 start with the same name as images in set1'''
    set1_stripped = [os.path.splitext(os.path.basename(n))[0] for n in set1]
    set2_stripped = [os.path.splitext(os.path.basename(n))[0] for n in set2]
    matches = [b.startswith(a) for a,b in zip(set1_stripped, set2_stripped)]
    return all(matches)

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

def read_metashape_poses(file):
    img_names = []
    #pose_array = np.zeros([0,4,4])
    with open(file) as f: 
        first_line = f.readline()
        if not first_line.startswith('Image_name,4x4 Tmatrix as 1x16 row'):
            raise ValueError("File doesn't start with 'Image_name,4x4 Tmatrix as 1x16 row' might be wrong format")
        data = f.readlines()
        pose_array = np.zeros([len(data),4,4])
        for i,line in enumerate(data):
            name, T_string = (line.strip().split(',',maxsplit=1))
            T = np.fromstring(T_string,sep=',').reshape((4,4))
            img_names.append(name)
            pose_array[i] = T
    return img_names, pose_array

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = '/home/vik748/data'
    
sets_folder = 'feature_descriptor_comparision'
test_set = 'set_1'

img_folder = os.path.join(path,sets_folder,test_set)

raw_images = sorted(glob.glob(img_folder+'/*.JPG'))
clahe_images = sorted(glob.glob(img_folder+'/*.tif'))
poses_txt = os.path.join(path,sets_folder,test_set,'poses.txt')

assert match_image_names(raw_images, clahe_images), "Images names of raw and clahe_images don't match"
assert len(raw_images) == 2, "Number of images in set is not 2 per type"


img = cv2.imread(path+'data/kitti/00/image_0/000000.png',0) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img = cv2.imread(path+'data/test_set/GOPR1429.JPG',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

    
gray = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create a mask image filled with zeros, the size of original image

tiley = 17
tilex = 4
total_feat = 25000

# defailts: int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, 
# int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, 
# int patchSize=31, int fastThreshold=20)

orb_detector = cv2.ORB_create(nfeatures=1000, edgeThreshold=125, patchSize=125, nlevels=8, 
                     fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike_detector = MultiHarrisZernike(Nfeats=600,like_matlab=True)


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
