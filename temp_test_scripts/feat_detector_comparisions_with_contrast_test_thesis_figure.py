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
sys.path.insert(0, os.path.abspath('../low_contrast_feature_detector_comparisons/'))
sys.path.insert(0, os.path.abspath('..'))
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features, draw_arrows
from feat_detector_comparisions_helper import *
import progressbar
from datetime import datetime

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('~/data/')

raw_sets_folder = 'Lars1_080818_800x600'
clahe_sets_folder = 'Lars1_080818_clahe_800x600'

TILE_KP = False
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
#K = np.array([[3523.90252470728501/5, 0.0, 2018.22833167806152/5],
#              [0.0, 3569.92180686745451/5, 1473.25249541175890/5],
#              [0.0, 0.0, 1.0]])

#D = np.array([-2.81360302828763176e-01, 1.38000456840603303e-01, 4.87629635176304053e-05, -6.01560125682630380e-05, -4.34666626743886730e-02])

raw_img_folder = os.path.join(path,raw_sets_folder)
clahe_img_folder = os.path.join(path,clahe_sets_folder)

raw_image_names = sorted(glob.glob(raw_img_folder+'/*.png'))[:25]
clahe_image_names = sorted(glob.glob(clahe_img_folder+'/*.tif'))[:25]
#poses_txt = os.path.join(path,sets_folder,test_set,'poses.txt')

#assert match_image_names(raw_image_names, clahe_image_names), "Images names of raw and clahe_images don't match"
#assert len(raw_image_names) == 2, "Number of images in set is not 2 per type"

'''
Detect Features
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

#criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
KLT_optical_flow = cv2.SparsePyrLKOpticalFlow_create(crit= (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 0.01),
                                                     maxLevel= 4, winSize= (25,25), minEigThreshold= 1e-3)

config_settings = {'set_title': 'Lars1 800x600 Raw Images', #'K':K, 'D':D,
                   'TILE_KP':TILE_KP, 'tiling':tiling ,
                   'detector': zernike, 'descriptor': orb, 'findFundamentalMat_params':findFundamentalMat_params}

#img0_name = raw_image_names[2]
#img1_name = raw_image_names[16]

img0_name = '/home/vik748/data/Stingray2_test_images/4bit/G0035800_ctrst_adj_-1.00.png'
img1_name = '/home/vik748/data/Stingray2_test_images/4bit/G0035820_ctrst_adj_-1.00.png'

image_0 = cv2.imread(img0_name, cv2.IMREAD_GRAYSCALE)
image_1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)

#results = analyze_image_pair(image_0, image_1, config_settings, plotMatches = True)


#K = settings['K']
#D = settings['D']
settings = config_settings
plotMatches = True
saveFig = False

TILE_KP = settings['TILE_KP']
tiling = settings['tiling']
detector = settings['detector']
descriptor = settings['descriptor']
det_name = detector.getDefaultName().replace('Feature2D.','')
des_name = descriptor.getDefaultName().replace('Feature2D.','')
if isinstance(descriptor, cv2.SparsePyrLKOpticalFlow): des_name='SparsePyrLKOpticalFlow'

if detector == descriptor and not TILE_KP:
    kp_0, des_0 = detector.detectAndCompute(image_0, mask=None)
    kp_1, des_1 = detector.detectAndCompute(image_1, mask=None)
else:
    detector_kp_0 = detector.detect(image_0, mask=None)
    detector_kp_1 = detector.detect(image_1, mask=None)

    if TILE_KP:
        kp_0 = tiled_features(detector_kp_0, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        kp_1 = tiled_features(detector_kp_1, image_1.shape, tiling[0], tiling[1], no_features= 1000)

    else:
        kp_0 = detector_kp_0
        kp_1 = detector_kp_1

    if not isinstance(descriptor, cv2.SparsePyrLKOpticalFlow):
        kp_0, des_0 = descriptor.compute(image_0, kp_0)
        kp_1, des_1 = descriptor.compute(image_1, kp_1)

if plotMatches:
    det_des_string = "Det: {} Des: {}".format(det_name, des_name)
    kp_img_0 = image_0
    kp_img_1 = image_1

    if TILE_KP:
        feat_string_0 = "{}\nBefore tiling:{:d} after tiling {:d}".format(det_des_string, len(detector_kp_0), len(kp_0))
        feat_string_1 = "Before tiling:{:d} after tiling {:d}".format(len(detector_kp_1), len(kp_1))

        kp_img_0 = draw_markers(kp_img_0, detector_kp_0, color=[255,255,0])
        kp_img_1 = draw_markers(kp_img_1, detector_kp_1, color=[255,255,0])

    else:
        feat_string_0 = "{}\n{:d} features".format(det_des_string, len(kp_0))
        feat_string_1 = "{:d} features".format(len(kp_1))

    kp_img_0 = draw_markers(kp_img_0, kp_0, color=[0,255,0])
    kp_img_1 = draw_markers(kp_img_1, kp_1, color=[0,255,0])

    #fig1 = plt.figure(1); plt.clf()
    fig1, fig1_axes = plt.subplots(2,1)
    fig1.suptitle(settings['set_title'] + ' features')
    fig1_axes[0].axis("off"); fig1_axes[0].set_title(feat_string_0)
    fig1_axes[0].imshow(kp_img_0)
    fig1_axes[1].axis("off"); fig1_axes[1].set_title(feat_string_1)
    fig1_axes[1].imshow(kp_img_1)
    fig1.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.1)
    plt.draw();

'''
Match and find inliers
'''




if isinstance(descriptor, cv2.SparsePyrLKOpticalFlow):
    kp0_match_01_full = cv2.KeyPoint_convert(kp_0)
    kp1_match_01_full, mask_matching, err = descriptor.calc(image_0, image_1, kp0_match_01_full, None)
    kp0_match_01 = kp0_match_01_full[mask_matching[:,0].astype(bool)]
    kp1_match_01 = kp1_match_01_full[mask_matching[:,0].astype(bool)]

else:
    if isinstance(descriptor, cv2.ORB) and descriptor.getWTA_K() != 2 :
        print ("ORB with WTA_K !=2")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=False)

    matches_01 = knn_match_and_lowe_ratio_filter(matcher, des_0, des_1, threshold=0.9)
    mask_matching = None

    kp0_match_01 = np.array([kp_0[mat.queryIdx].pt for mat in matches_01])
    kp1_match_01 = np.array([kp_1[mat.trainIdx].pt for mat in matches_01])

#kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(kp0_match_01,axis=1),K,D)
#kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(kp1_match_01,axis=1),K,D)

#E_12, mask_e_12 = cv2.findEssentialMat(kp0_match_01_ud, kp1_match_01_ud, focal=1.0, pp=(0., 0.),
#                                       method=cv2.RANSAC, prob=0.9999, threshold=0.001)

E_12, mask_e_12 = cv2.findFundamentalMat(kp0_match_01, kp1_match_01, mask=mask_matching, **settings['findFundamentalMat_params'])

no_matches = np.sum(mask_e_12)
result = {'detector':det_name, 'descriptor':des_name, 'matches': no_matches,
          'img0_no_features': len(kp_0), 'img1_no_features':len(kp_1)}

if plotMatches:
    #valid_matches_img = draw_matches_vertical(image_0, kp_0, image_1, kp_1, matches_01,
    #                                                  mask_e_12, display_invalid=True, color=(0, 255, 0))
    if isinstance(descriptor, cv2.SparsePyrLKOpticalFlow):
        valid_matches_img = draw_arrows(image_1, kp0_match_01[mask_e_12[:,0]==1], kp1_match_01[mask_e_12[:,0]==1], color=(0, 255, 0), thick = 2)
    else:
        valid_matches_img = draw_feature_tracks(image_0, kp_0, image_1, kp_1, matches_01,
                                                mask_e_12, display_invalid=True, color=(0, 255, 0),
                                                thick = 2)

    #fig2 = plt.figure(2); plt.clf()
    fig2, fig2_axes = plt.subplots(1,1)
    fig2.suptitle(settings['set_title'] + ' Feature Matching')
    fig2_axes.axis("off"); fig2_axes.set_title("{}\n{:d} matches".format(det_des_string, no_matches))
    fig2_axes.imshow(valid_matches_img)
    fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)
    plt.draw(); plt.show(block=False); plt.pause(.2)
    if saveFig:
        save_fig2png(fig2)
    #input("Enter to continue")