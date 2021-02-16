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

hostname = os.uname().nodename

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
elif hostname=='NEUFR-TP02':
    path = os.path.expanduser('~/data/')
else:
    path = os.path.expanduser('/data/')

datasets = ['Lars1_080818_800x600', 'Lars2_081018_800x600']

image_names = { 'Lars1_080818_800x600': ['G0285493.png', 'G0285513.png'], # Lars1
                'Lars2_081018_800x600': ['G0028388.JPG.png', 'G0028408.JPG.png'] }# Lars2

dataset_name = datasets[1]

img0_name = os.path.join(path, dataset_name, image_names[dataset_name][0] )
img1_name = os.path.join(path, dataset_name, image_names[dataset_name][1] )

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

zernike = MultiHarrisZernike(Nfeats= NO_OF_FEATURES, seci= 2, secj= 3, levels= 6, ratio= 1/1.2,
                             sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3)

surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 50, nOctaves = 6)

sift = cv2.xfeatures2d.SIFT_create(nfeatures = NO_OF_UT_FEATURES, nOctaveLayers = 3, contrastThreshold = 0.01,
                                   edgeThreshold = 20, sigma = 1.6)

findFundamentalMat_params = {'method':cv2.FM_RANSAC,       # RAnsac
                             'param1':1.0,                # Inlier threshold in pixel since we don't use nomalized coordinates
                             'param2':0.9999}              # Confidence level

KLT_optical_flow = cv2.SparsePyrLKOpticalFlow_create(crit= (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 0.01),
                                                     maxLevel= 4, winSize= (25,25), minEigThreshold= 1e-3)

config_settings = {'set_title': dataset_name, #'K':K, 'D':D, 
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': surf, 'descriptor': surf, 'findFundamentalMat_params':findFundamentalMat_params}

image_0 = read_grimage(img0_name)
image_1 = read_grimage(img1_name)

#results = analyze_image_pair(image_0, image_1, config_settings, plotMatches = True)

settings = config_settings
plotMatches = True
saveFig = False

TILE_KP = settings['TILE_KP']
tiling = settings['tiling']
detector = settings['detector']
descriptor = settings['descriptor']

det_name = detector.getDefaultName().replace('Feature2D.','')
if det_name == 'Feature2D':
    det_name = type(detector).__name__.replace('xfeatures2d_','')
des_name = descriptor.getDefaultName().replace('Feature2D.','')
if des_name == 'Feature2D':
    des_name = type(descriptor).__name__.replace('xfeatures2d_','')
    
if isinstance(descriptor, cv2.SparsePyrLKOpticalFlow): des_name='SparsePyrLKOpticalFlow'
    
if detector == descriptor and not TILE_KP:
    kp_0, des_0 = detector.detectAndCompute(image_0, mask=None)
    #kp_1, des_1 = detector.detectAndCompute(image_1, mask=None)
else:
    detector_kp_0 = detector.detect(image_0, mask=None)
    detector_kp_1 = detector.detect(image_1, mask=None)

    if TILE_KP:
        kp_0 = tiled_features(detector_kp_0, image_0.shape, tiling[0], tiling[1], no_features= NO_OF_FEATURES)
                
    else:
        kp_0 = detector_kp_0
        #kp_1 = detector_kp_1

    if not isinstance(descriptor, cv2.SparsePyrLKOpticalFlow):
        kp_0, des_0 = descriptor.compute(image_0, kp_0)
        #kp_1, des_1 = descriptor.compute(image_1, kp_1)
   

if isinstance(descriptor, cv2.ORB) and descriptor.getWTA_K() != 2 :
    print ("ORB with WTA_K !=2")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
else:
    matcher = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=False)

'''
Match and find second distances
'''

matches_00 = matcher.knnMatch(des_0, des_0, k=2, mask=None)
matches_00_second = [m[1] for m in matches_00]
second_distances = np.array([m[1].distance for m in matches_00])
marker_colors = normalize_and_applycolormap(second_distances, colormap = cv2.COLORMAP_RAINBOW)

#no_matches = np.sum(mask_e_12)
x = np.linspace(0,second_distances.max(),51)
sec_dist_kde_full = st.gaussian_kde(second_distances,bw_method='silverman')
sec_dist_kde = sec_dist_kde_full(x)
    
if plotMatches:        
    full_map_BGR = cv2.applyColorMap(np.linspace(0,255,256).astype(np.uint8),cv2.COLORMAP_RAINBOW)
    full_map_RGB = cv2.cvtColor(full_map_BGR, cv2.COLOR_BGR2RGB)
    full_map = full_map_RGB[:,0,:] /255
    cmap= mpl.colors.ListedColormap(full_map)
    norm= mpl.colors.Normalize(vmin=0,vmax=second_distances.max())
    
    det_des_string = "Det: {} Des: {}".format(det_name, des_name)
    kp_img_0 = image_0
    
    if TILE_KP:
        feat_string_0 = "{}\nBefore tiling:{:d} after tiling {:d}".format(det_des_string, len(detector_kp_0), len(kp_0))
#        kp_img_0 = draw_markers(kp_img_0, detector_kp_0, color=marker_colors, markerSize=1)
        
    else:            
        feat_string_0 = "{}\n{:d} features".format(det_des_string, len(kp_0))

    kp_img_0 = draw_markers(kp_img_0, kp_0, color=marker_colors)

    #fig1 = plt.figure(1); plt.clf()
    fig1, fig1_axes = plt.subplots(2,1)
    fig1.suptitle(settings['set_title'] + ' features')
    fig1_axes[0].axis("off"); fig1_axes[0].set_title(feat_string_0)
    fig1_axes[0].imshow(cv2.cvtColor(kp_img_0, cv2.COLOR_BGR2RGB))
    #cb = matplotlib.colorbar.ColorbarBase(fig1_axes[2], orientation='horizontal', 
    #                           cmap=cmap, norm=norm, fraction=1)
    
    fig1_axes[1].hist(second_distances, bins=x, color='blue', density=True, alpha=0.4, label='Raw')
    fig1_axes[1].fill_between(x, sec_dist_kde, color='red',alpha=0.4)
    fig1_axes[1].set_xlim((0,second_distances.max()))

    cb_ax = fig1.add_axes([0.125, 0.05, 0.9-0.125, 0.025])
    cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap,
                                   norm=norm, orientation='horizontal')

    if saveFig:
        save_fig2png(fig, size = [9.28, 9.58])

result = {'detector':det_name, 'descriptor':des_name, 'matches': 0, 
          'second_dist_kde': sec_dist_kde, 'img0_no_features': len(kp_0)}
