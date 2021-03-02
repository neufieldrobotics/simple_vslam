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
sys.path.insert(0, os.path.abspath('./external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
sys.path.insert(0, os.path.abspath('./external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
#sys.path.insert(0, os.path.abspath('../'))
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features, draw_arrows
from feat_detector_comparisions_helper import *
import progressbar
from datetime import datetime
import scipy.stats as st

hostname = os.uname().nodename

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
elif hostname=='NEUFR-TP02':
    path = os.path.expanduser('~/data/low_contrast_datasets')
else:
    path = os.path.expanduser('/data/low_contrast_datasets')

datasets = ['Lars1_080818_800x600', 'Lars2_081018_800x600', 'skerki_full', 'skerki_mud_CLAHE']

contrast_types = ['RAW', '7_bit', '6_bit', '5_bit', '4_bit', '3_bit']

image_names = { 'Lars1_080818_800x600': 'G0285493.png', #,'G0285513.png'], # Lars1
                'Lars2_081018_800x600': 'G0028388.JPG.png', #, 'G0028408.JPG.png'], 
                'skerki_full'         : 'ESC.970622_024806.0590.tif',
                'skerki_mud_CLAHE'    : 'ESC.970622_024806.0590.tif'}


dataset_name = datasets[3]
contrast_type = contrast_types[0]
img0_name = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, image_names[dataset_name] )

TILE_KP = True
tiling = (4,3)
NO_OF_FEATURES = 600
BASELINE_STEP_SIZE = 10


if TILE_KP:
    NO_OF_UT_FEATURES = NO_OF_FEATURES * 2
else:
    NO_OF_UT_FEATURES = NO_OF_FEATURES


'''
LOAD DATA
'''


'''
Setup Feature Detectors
'''
orb = cv2.ORB_create(nfeatures = NO_OF_UT_FEATURES, edgeThreshold=31, patchSize=31, nlevels=6,
                      fastThreshold=1, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike = MultiHarrisZernike(Nfeats= NO_OF_FEATURES, seci= 3, secj= 4, levels= 6, ratio= 1/1.2,
                             sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3)

surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 15, nOctaves = 6)

sift = cv2.xfeatures2d.SIFT_create(nfeatures = NO_OF_UT_FEATURES, nOctaveLayers = 6, contrastThreshold = 0.001,
                                   edgeThreshold = 20, sigma = 1.6)

findFundamentalMat_params = {'method':cv2.FM_RANSAC,       # RAnsac
                             'param1':1.0,                # Inlier threshold in pixel since we don't use nomalized coordinates
                             'param2':0.9999}              # Confidence level

KLT_optical_flow = cv2.SparsePyrLKOpticalFlow_create(crit= (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 0.01),
                                                     maxLevel= 4, winSize= (25,25), minEigThreshold= 1e-3)

# Plot settings database
plot_display_settings_database= {('ESC.970622_024806.0590.tif', orb):{'max_second_dist': 100, 'max_prob':0.04}, 
                                 ('ESC.970622_024806.0590.tif', zernike):{'max_second_dist': 9, 'max_prob':0.65},
                                 ('ESC.970622_024806.0590.tif', sift):{'max_second_dist': 400, 'max_prob':0.015},
                                 ('ESC.970622_024806.0590.tif', surf):{'max_second_dist': 0.7, 'max_prob':10}}


config_settings = {'set_title': dataset_name+"_"+contrast_type, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': orb, 'descriptor': orb, 'findFundamentalMat_params':findFundamentalMat_params,
                   'NO_OF_FEATURES': NO_OF_FEATURES }

plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
if plot_display_settings is not None: 
    config_settings.update(plot_display_settings)

image_0 = read_grimage(img0_name)

analyze_descriptor_distance_image_pair(image_0, config_settings, plotMatches=True, saveFig=True)