#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 HISTORY      WHO     WHAT
----------    ----    -------------------------------------
2021-02-14    vs      Started writing this file using 'feat_detector_comparisions_with_contrast_test' as a template
                      The goal of this file is to write a single pair feature descriptor distance analysis function
"""
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import glob
sys.path.insert(0, os.path.abspath('../external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
sys.path.insert(0, os.path.abspath('../external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
sys.path.insert(0, os.path.abspath('../'))
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features, draw_arrows
from feat_detector_comparisions_helper import *
import progressbar
from datetime import datetime
import scipy.stats as st
from feat_descriptor_comparison_setup import *

dataset_name = datasets[2]
contrast_type = contrast_types[0]

img0_name = image_names[dataset_name]
img0_name_base, image0_name_ext = os.path.splitext(img0_name)
img0_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
img0_mask_filename = os.path.join(path, dataset_name, 'masks', img0_name_base+'_mask'+image0_name_ext )

'''
LOAD DATA
'''



#img0_name = "/home/vik748/data/goprocalib_80.75mm_target_set_2_800x600/GOPR1510.png"
#image_0 = read_grimage(img0_filename)
#image_0_mask = read_grimage(img0_mask_filename)
dataset_name = "White Noise"
image_0 = (np.round(np.random.rand(600,800)*255)).astype(np.uint8)
image_0_mask = None


config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': zernike, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES}

#plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
#if plot_display_settings is not None: 
#    config_settings.update(plot_display_settings)

results = analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                                 mask = image_0_mask, plotMatches=True, saveFig=True)


config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': sift, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES}
                   #'provided_keypoints': results['keypoints']}

results = analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                                 plotMatches=True, saveFig=True)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': orb, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES}
                   #'provided_keypoints': results['keypoints']}

results = analyze_descriptor_distance_image_pair(image_0, config_settings, plotMatches=True, saveFig=True)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': surf, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES}
                   #'provided_keypoints': results['keypoints']}

results = analyze_descriptor_distance_image_pair(image_0, config_settings, plotMatches=True, saveFig=True)
