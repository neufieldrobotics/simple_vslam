#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 HISTORY      WHO     WHAT
----------    ----    -------------------------------------
2021-02-25    vs      Started writing this file using 'feat_detector_comparisions_with_contrast_test' as a template
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

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

dataset_name = datasets[7]
contrast_type = contrast_types[0]

img0_name = image_names[dataset_name]
img0_name_base, image0_name_ext = os.path.splitext(img0_name)
img0_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
img0_mask_filename = os.path.join(path, dataset_name, 'masks', img0_name_base+'_mask'+image0_name_ext )

image_0 = read_grimage(img0_filename)
image_0_mask = read_grimage(img0_mask_filename)

#_, _, kp_eig_vals, Ft = zernike.detectAndCompute(image_0, computeEigVals=True, mask=image_0_mask, timing=False)
#kp_sift, des = zernike.detectAndCompute(image_0, computeEigVals=False, mask=image_0_mask, timing=False)
kp_sift, des = sift.detectAndCompute(image_0, mask=image_0_mask)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': sift, 'descriptor': sift, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES,
                   #'provided_keypoints': kp_sift,
                   #'kp_eigen_values':kp_eig_vals, 'provided_Ft': Ft,
                   'plot_eig_movement': False}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)

results = analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                                 mask = image_0_mask, plotMatches=True, saveFig=True)

kp_zernike, des = zernike.detectAndCompute(image_0, mask=image_0_mask)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': sift, 'descriptor': sift, 'findFundamentalMat_params':findFundamentalMat_params,
                   #'NO_OF_FEATURES': NO_OF_FEATURES}
                   'provided_keypoints': kp_zernike}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)


analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                       plotMatches=True, saveFig=True)

kp_sift_res = results['keypoints'].copy()
kp_sift_o2s =[]
for k in kp_sift_res:
    octave, layer, scale = unpackOctave(k)
    k.octave = layer-1
    # k.octave = layer -1 + int(np.log2(scale))
    kp_sift_o2s.append(k)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': zernike, 'findFundamentalMat_params':findFundamentalMat_params,
                   #'NO_OF_FEATURES': NO_OF_FEATURES}
                   'provided_keypoints': kp_sift_o2s}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)


analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                       plotMatches=True, saveFig=True)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': zernike, 'findFundamentalMat_params':findFundamentalMat_params,
                   #'NO_OF_FEATURES': NO_OF_FEATURES}
                   'provided_keypoints': kp_zernike}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)


analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                       plotMatches=True, saveFig=True)


dataset_name = datasets[8]
contrast_type = contrast_types[0]

img0_name = image_names[dataset_name]
img0_name_base, image0_name_ext = os.path.splitext(img0_name)
img0_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
img0_mask_filename = os.path.join(path, dataset_name, 'masks', img0_name_base+'_mask'+image0_name_ext )

image_0 = read_grimage(img0_filename)
image_0_mask = read_grimage(img0_mask_filename)

kp_sift_res = results['keypoints'].copy()

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': sift, 'descriptor': sift, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES,
                   'provided_keypoints': kp_sift_res,
                   #'kp_eigen_values':kp_eig_vals, 'provided_Ft': Ft,
                   'plot_eig_movement': False}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)

analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                                 mask = image_0_mask, plotMatches=True, saveFig=True)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': sift, 'descriptor': sift, 'findFundamentalMat_params':findFundamentalMat_params,
                   #'NO_OF_FEATURES': NO_OF_FEATURES}
                   'provided_keypoints': kp_zernike}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)


analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                       plotMatches=True, saveFig=True)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': zernike, 'findFundamentalMat_params':findFundamentalMat_params,
                   #'NO_OF_FEATURES': NO_OF_FEATURES}
                   'provided_keypoints': kp_sift_o2s}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)


analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                       plotMatches=True, saveFig=True)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': zernike, 'findFundamentalMat_params':findFundamentalMat_params,
                   #'NO_OF_FEATURES': NO_OF_FEATURES}
                   'provided_keypoints': kp_zernike}

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)


analyze_descriptor_distance_image_pair(image_0, config_settings, 
                                       plotMatches=True, saveFig=True)