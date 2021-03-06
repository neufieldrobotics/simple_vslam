#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 HISTORY      WHO     WHAT
----------    ----    -------------------------------------
2021-02-14    vs      feat_detector_comparisions_setup is used to keep all the basic
                      setup variables in one place so that the other scripts can use them
"""
import numpy as np
import cv2
import sys
#from matplotlib import pyplot as plt
#import matplotlib as mpl
import os
import glob
sys.path.insert(0, os.path.abspath('../external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
#sys.path.insert(0, os.path.abspath('../external_packages/cmtpy/'))
#from cmtpy.histogram_warping_ace import HistogramWarpingACE
#sys.path.insert(0, os.path.abspath('../'))
#from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features, draw_arrows
#from feat_detector_comparisions_helper import *
#import progressbar
#from datetime import datetime
#import scipy.stats as st

hostname = os.uname().nodename

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
elif hostname=='NEUFR-TP02':
    path = os.path.expanduser('~/data/low_contrast_datasets')
else:
    path = os.path.expanduser('/data/low_contrast_datasets')

contrast_types = ['RAW', '7_bit', '6_bit', '5_bit', '4_bit', '3_bit']

image_names = { 'Lars1_080818_800x600'      : 'G0285493.png', #,'G0285513.png'], # Lars1
                'Lars2_081018_800x600'      : 'G0028388.png', #, 'G0028408.JPG.png'], 
                'skerki_full'               : 'ESC.970622_030232.0655.tif',
                'skerki_mud'                : 'ESC.970622_024806.0590.tif',
                'skerki_mud_CLAHE'          : 'ESC.970622_024806.0590.tif',
                'lab_with_target'           : 'GOPR1511.png',
                'Stingray2_080718_800x600'  : 'G0035780.png',
                'Morgan2_UAV_800x600'       : 'DJI_0875.png',
                'Morgan2_UAV_800x600_CLAHE' : 'DJI_0875.png'}

datasets = sorted(image_names.keys())

tiling = (4,3)
NO_OF_FEATURES = 100
BASELINE_STEP_SIZE = 10

TILE_KP = False

if TILE_KP:
    NO_OF_UT_FEATURES = NO_OF_FEATURES * 2
else:
    NO_OF_UT_FEATURES = NO_OF_FEATURES


'''
Setup Feature Detectors
'''
orb = cv2.ORB_create(nfeatures = NO_OF_UT_FEATURES, edgeThreshold=31, patchSize=31, nlevels=6,
                      fastThreshold=1, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike = MultiHarrisZernike(Nfeats= NO_OF_FEATURES, seci= 3, secj= 4, levels= 6, ratio= 1/1.2, #0.5, 
                             sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3, harris_threshold = None   )

surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 15, nOctaves = 1)

sift = cv2.xfeatures2d.SIFT_create(nfeatures = NO_OF_UT_FEATURES, nOctaveLayers = 6, contrastThreshold = 0.001,
                                   edgeThreshold = 20, sigma = 1.6)

findFundamentalMat_params = {'method':cv2.FM_RANSAC,       # RAnsac
                             'param1':1.0,                # Inlier threshold in pixel since we don't use nomalized coordinates
                             'param2':0.9999}              # Confidence level

KLT_optical_flow = cv2.SparsePyrLKOpticalFlow_create(crit= (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 0.01),
                                                     maxLevel= 4, winSize= (25,25), minEigThreshold= 1e-3)

# Plot settings database
plot_display_settings_database= {('ESC.970622_024806.0590.tif', orb):{'max_second_dist': 100, 'max_prob':0.04}, 
                                 ('ESC.970622_024806.0590.tif', zernike):{'max_second_dist': 9, 'max_prob':0.8, 'eig_plot_lims': [67, 89 ]},
                                 ('ESC.970622_024806.0590.tif', sift):{'max_second_dist': 400, 'max_prob':0.015},
                                 ('ESC.970622_024806.0590.tif', surf):{'max_second_dist': 0.7, 'max_prob':10},
                                 ('G0285493.png', sift):{'max_second_dist': 450, 'max_prob':0.03},
                                 ('DJI_0875.png', zernike):{'max_second_dist': 9.22, 'max_prob':0.72, 'eig_plot_lims': [66, 62]},
                                 }

np.random.seed(7)
