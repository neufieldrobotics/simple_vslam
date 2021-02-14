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
sys.path.insert(0, os.path.abspath('../external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
sys.path.insert(0, os.path.abspath('../external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features, draw_arrows
from feat_detector_comparisions_helper import *
import progressbar
from datetime import datetime

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('/data/Stingray')

raw_sets_folder = 'Stingray2_080718_800x600'


if TILE_KP:
    NO_OF_UT_FEATURES = NO_OF_FEATURES * 2
else:
    NO_OF_UT_FEATURES = NO_OF_FEATURES

raw_img_folder = os.path.join(path,raw_sets_folder)

raw_image_names = sorted(glob.glob(raw_img_folder+'/*.png'))

image_names = raw_image_names
contrast_adj_factors = np.arange(0,-1.1,-.1)
contrast_img_folder = raw_img_folder + '_ctrst_imgs_3bit'

bit_depths = np.zeros((len(image_names),len(contrast_adj_factors)))

for i,img_name in enumerate(image_names):
    img_base, img_ext = os.path.splitext(os.path.basename(img_name))
    raw_sets_folder = os.path.basename(os.path.dirname(img_name))

    for j,ctrst_adj_fact in enumerate(contrast_adj_factors):
        ctrst_adj_img_basename = "{}_ctrst_adj_{:.2f}{}".format(img_base, ctrst_adj_fact,img_ext)
        ctrst_adj_img_name = os.path.join(contrast_img_folder, ctrst_adj_img_basename)        
        img = read_grimage(ctrst_adj_img_name)
        bit_depths[i,j] = np.max(img) - np.min(img)
        