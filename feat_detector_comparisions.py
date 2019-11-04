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
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features
from feat_detector_comparisions_helper import *


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = '/home/vik748/data'
    
sets_folder = 'feature_descriptor_comparision'
test_set = 'set_1'

TILE_KP = True
tiling = (4,3)
NO_OF_FEATURES = 600

if TILE_KP:
    NO_OF_UT_FEATURES = NO_OF_FEATURES * 2
else:
    NO_OF_UT_FEATURES = NO_OF_FEATURES

'''
LOAD DATA
'''
K = np.array([[3523.90252470728501/5, 0.0, 2018.22833167806152/5],
              [0.0, 3569.92180686745451/5, 1473.25249541175890/5],
              [0.0, 0.0, 1.0]])

D = np.array([-2.81360302828763176e-01, 1.38000456840603303e-01, 4.87629635176304053e-05, -6.01560125682630380e-05, -4.34666626743886730e-02])

img_folder = os.path.join(path,sets_folder,test_set)

raw_image_names = sorted(glob.glob(img_folder+'/*.JPG'))
clahe_image_names = sorted(glob.glob(img_folder+'/*.tif'))
poses_txt = os.path.join(path,sets_folder,test_set,'poses.txt')

assert match_image_names(raw_image_names, clahe_image_names), "Images names of raw and clahe_images don't match"
assert len(raw_image_names) == 2, "Number of images in set is not 2 per type"

'''
Detect Features
'''
orb_detector = cv2.ORB_create(nfeatures = NO_OF_UT_FEATURES, edgeThreshold=31, patchSize=31, nlevels=6, 
                              fastThreshold=10, scaleFactor=1.2, WTA_K=2,
                              scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike_detector = MultiHarrisZernike(Nfeats= NO_OF_FEATURES, seci= 2, secj= 3, levels= 6, ratio= 1/1.2, 
                                      sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3)

sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures = NO_OF_UT_FEATURES, nOctaveLayers = 3, contrastThreshold = 0.01, 
                                            edgeThreshold = 20, sigma = 1.6)

raw_images = read_image_list(raw_image_names, resize_ratio=1/5)
clahe_images = read_image_list(clahe_image_names, resize_ratio=1/5)

image_0 = raw_images[0]
image_1 = raw_images[1]

config_settings = {'set_title': 'Lars1 800x600 Raw Images',
                   'K':K, 'D':D, 'TILE_KP':TILE_KP, 'tiling':tiling , 
                   'zernike_detector': zernike_detector, 'orb_detector': orb_detector, 'sift_detector': sift_detector} 

analyze_image_pair(image_0, image_1, config_settings)

