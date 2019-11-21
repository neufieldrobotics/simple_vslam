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
from matlab_imresize.imresize import imresize
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features

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

def read_image_list(img_names, resize_ratio=1):
    images = []
    for name in img_names:
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        if resize_ratio != 1:
            img = imresize(img, resize_ratio, method='bicubic')
        images.append(img)
        
    return images

def draw_matches_vertical(img_top, kp1,img_bottom,kp2, matches, mask, display_invalid=False, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    assert img_top.shape == img_bottom.shape
    out_img = np.vstack((img_top, img_bottom))
    bool_mask = mask.astype(bool)
    valid_bottom_matches = np.array([kp2[mat.trainIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    valid_top_matches = np.array([kp1[mat.queryIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    img_height = img_top.shape[0]

    if len(out_img.shape) == 2: out_img = cv2.cvtColor(out_img,cv2.COLOR_GRAY2RGB)

    for p1,p2 in zip(valid_top_matches, valid_bottom_matches):
        cv2.line(out_img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1]+img_height)), color=color, thickness=1)
    return out_img


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = '/home/vik748/data'
    
sets_folder = 'feature_descriptor_comparision'
test_set = 'set_1'

TILE_KP = True
tiling = (4,3)
NO_OF_FEATURES = 600

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
orb_detector = cv2.ORB_create(nfeatures=2 * NO_OF_FEATURES, edgeThreshold=31, patchSize=31, nlevels=6, 
                              fastThreshold=1, scaleFactor=1.2, WTA_K=2,
                              scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike_detector = MultiHarrisZernike(Nfeats= NO_OF_FEATURES, seci= 2, secj= 3, levels= 6, ratio= 1/1.2, 
                                      sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3)

sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures = 2 * NO_OF_FEATURES, nOctaveLayers = 3, contrastThreshold = 0.01, 
                                            edgeThreshold = 20, sigma = 1.6)

surf_detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 50, nOctaves = 6)

raw_images = read_image_list(raw_image_names, resize_ratio=1/5)
clahe_images = read_image_list(clahe_image_names, resize_ratio=1/5)

zernike_kp_0, zernike_des_0 = zernike_detector.detectAndCompute(raw_images[0], mask=None, timing=False)
zernike_kp_1, zernike_des_1 = zernike_detector.detectAndCompute(raw_images[1], mask=None, timing=False)
orb_kp_0 = orb_detector.detect(raw_images[0], None)
orb_kp_1 = orb_detector.detect(raw_images[1], None)
sift_kp_0, sift_des_0 = sift_detector.detectAndCompute(raw_images[0], None)
sift_kp_1, sift_des_1 = sift_detector.detectAndCompute(raw_images[1], None)
surf_kp_0 = surf_detector.detect(raw_images[0], None)
surf_kp_1 = surf_detector.detect(raw_images[1], None)

print ("Points before tiling supression: ",len(surf_kp_0))

if TILE_KP:
    orb_kp_0 = tiled_features(orb_kp_0, raw_images[0].shape, tiling[0], tiling[1], no_features= 1000)
    orb_kp_1 = tiled_features(orb_kp_1, raw_images[1].shape, tiling[0], tiling[1], no_features= 1000)
    surf_kp_0 = tiled_features(surf_kp_0, raw_images[0].shape, tiling[0], tiling[1], no_features= 1000)
    surf_kp_1 = tiled_features(surf_kp_1, raw_images[1].shape, tiling[0], tiling[1], no_features= 1000)

    print ("Points after tiling supression: ",len(surf_kp_0))

orb_kp_0, orb_des_0 = orb_detector.compute(raw_images[0], surf_kp_0)
orb_kp_1, orb_des_1 = orb_detector.compute(raw_images[1], surf_kp_1)
surf_kp_0, surf_des_0 = surf_detector.compute(raw_images[0], surf_kp_0)
surf_kp_1, surf_des_1 = surf_detector.compute(raw_images[1], surf_kp_1)

zernike_kp_0_sort = sorted(zernike_kp_0, key = lambda x: x.response, reverse=True)
zernike_kp_1_sort = sorted(zernike_kp_1, key = lambda x: x.response, reverse=True)
orb_kp_0_sort = sorted(orb_kp_0, key = lambda x: x.response, reverse=True)
orb_kp_1_sort = sorted(orb_kp_1, key = lambda x: x.response, reverse=True)
sift_kp_0_sort = sorted(sift_kp_0, key = lambda x: x.response, reverse=True)
sift_kp_1_sort = sorted(sift_kp_1, key = lambda x: x.response, reverse=True)

zernike_kp_img_0 = draw_markers (cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB), zernike_kp_0 ,color=[255,255,0])
zernike_kp_img_1 = draw_markers (cv2.cvtColor(raw_images[1], cv2.COLOR_GRAY2RGB), zernike_kp_1 ,color=[255,255,0])
orb_kp_img_0 = draw_markers(cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB), orb_kp_0, color=[255,255,0])
orb_kp_img_1 = draw_markers(cv2.cvtColor(raw_images[1], cv2.COLOR_GRAY2RGB), orb_kp_1, color=[255,255,0])
surf_kp_img_0 = draw_markers(cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB), surf_kp_0,color=[255,255,0])
surf_kp_img_1 = draw_markers(cv2.cvtColor(raw_images[1], cv2.COLOR_GRAY2RGB), surf_kp_1,color=[255,255,0])

    
fig1, fig1_axes = plt.subplots(2,3)
fig1.suptitle('800x600 Raw Images Top 25 features')
fig1_axes[0,0].axis("off"); fig1_axes[0,0].set_title("Zernike Features")
fig1_axes[0,0].imshow(zernike_kp_img_0)
fig1_axes[1,0].axis("off")
fig1_axes[1,0].imshow(zernike_kp_img_1)
fig1_axes[0,1].axis("off")
fig1_axes[0,1].imshow(orb_kp_img_0)
fig1_axes[1,1].axis("off"); fig1_axes[0,1].set_title("Orb Features")
fig1_axes[1,1].imshow(orb_kp_img_1)
fig1_axes[0,2].axis("off")
fig1_axes[0,2].imshow(surf_kp_img_0)
fig1_axes[1,2].axis("off"); fig1_axes[0,2].set_title("SURF Features")
fig1_axes[1,2].imshow(surf_kp_img_1)
#fig1.subplots_adjust(0,0,1,1,0.0,0.0)
fig1.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)
plt.show()

'''
Match and find inliers
'''
matcher_norm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matcher_hamming = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)

zernike_matches_01 = knn_match_and_lowe_ratio_filter(matcher_norm, zernike_des_0, zernike_des_1, threshold=0.9)

zernike_kp0_match_01 = np.array([zernike_kp_0[mat.queryIdx].pt for mat in zernike_matches_01])
zernike_kp1_match_01 = np.array([zernike_kp_1[mat.trainIdx].pt for mat in zernike_matches_01])

zernike_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(zernike_kp0_match_01,axis=1),K,D)
zernike_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(zernike_kp1_match_01,axis=1),K,D)

zernike_E_12, zernike_mask_e_12 = cv2.findEssentialMat(zernike_kp0_match_01_ud, zernike_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                       method=cv2.RANSAC, prob=0.9999, threshold=0.001)

print("Zernike After essential: ", np.sum(zernike_mask_e_12))

zernike_valid_matches_img = draw_matches_vertical(raw_images[0],zernike_kp_0, raw_images[1],zernike_kp_1, zernike_matches_01, 
                                              zernike_mask_e_12, display_invalid=True, color=(0, 255, 0))


orb_matches_01 = knn_match_and_lowe_ratio_filter(matcher_hamming, orb_des_0, orb_des_1, threshold=0.9)

orb_kp0_match_01 = np.array([orb_kp_0[mat.queryIdx].pt for mat in orb_matches_01])
orb_kp1_match_01 = np.array([orb_kp_1[mat.trainIdx].pt for mat in orb_matches_01])

orb_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp0_match_01,axis=1),K,D)
orb_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp1_match_01,axis=1),K,D)

orb_E_12, orb_mask_e_12 = cv2.findEssentialMat(orb_kp0_match_01_ud, orb_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                               method=cv2.RANSAC, prob=0.9999, threshold=0.001)

print("Orb After essential: ", np.sum(orb_mask_e_12))

orb_valid_matches_img = draw_matches_vertical(raw_images[0],orb_kp_0, raw_images[1],orb_kp_1, orb_matches_01, 
                                              orb_mask_e_12, display_invalid=True, color=(0, 255, 0))


surf_matches_01 = knn_match_and_lowe_ratio_filter(matcher_norm, surf_des_0, surf_des_1, threshold=0.90)

surf_kp0_match_01 = np.array([surf_kp_0[mat.queryIdx].pt for mat in surf_matches_01])
surf_kp1_match_01 = np.array([surf_kp_1[mat.trainIdx].pt for mat in surf_matches_01])

surf_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(surf_kp0_match_01,axis=1),K,D)
surf_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(surf_kp1_match_01,axis=1),K,D)

surf_E_12, surf_mask_e_12 = cv2.findEssentialMat(surf_kp0_match_01_ud, surf_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                               method=cv2.RANSAC, prob=0.9999, threshold=0.001)

print("surf After essential: ", np.sum(surf_mask_e_12))

surf_valid_matches_img = draw_matches_vertical(raw_images[0],surf_kp_0, raw_images[1],surf_kp_1, surf_matches_01, 
                                              surf_mask_e_12, display_invalid=True, color=(0, 255, 0))


fig2, fig2_axes = plt.subplots(1,3)
fig2.suptitle('800x600 Raw Images Feature Matching')
fig2_axes[0].axis("off"); fig2_axes[0].set_title("Zernike Features\n{:d} matches".format(np.sum(zernike_mask_e_12)))
fig2_axes[0].imshow(zernike_valid_matches_img)
fig2_axes[1].axis("off"); fig2_axes[1].set_title("Orb Features\n{:d} matches".format(np.sum(orb_mask_e_12)))
fig2_axes[1].imshow(orb_valid_matches_img)
fig2_axes[2].axis("off"); fig2_axes[2].set_title("surf Features\n{:d} matches".format(np.sum(surf_mask_e_12)))
fig2_axes[2].imshow(surf_valid_matches_img)
fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)


points, rot_2R1, trans_2t1, mask_RP_12 = cv2.recoverPose(zernike_E_12, 
                                                         orb_kp0_match_01_ud, 
                                                         orb_kp1_match_01_ud,
                                                         mask=orb_mask_e_12)

imageSize = (raw_images[0].shape[1], raw_images[0].shape[0])
(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                K, D, K, D, imageSize, rot_2R1.T, -trans_2t1,
                None, None, None, None, None,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(K, D, leftRectification, leftProjection, imageSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K, D, rightRectification, rightProjection, imageSize, cv2.CV_32FC1)

fixedLeft = cv2.remap(raw_images[0], leftMapX, leftMapY, cv2.INTER_CUBIC)
fixedRight = cv2.remap(raw_images[1], rightMapX, rightMapY, cv2.INTER_CUBIC)

fig3, fig3_axes = plt.subplots(1,2)
fig3.suptitle('800x600 Raw Images Rectified Images')
fig3_axes[0].axis("off"); fig3_axes[0].set_title("Last image")
fig3_axes[0].imshow(fixedLeft, cmap='gray')
fig3_axes[1].axis("off"); fig3_axes[1].set_title("Current Image")
fig3_axes[1].imshow(fixedRight, cmap='gray')

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(15)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(200)


depth = stereoMatcher.compute(fixedLeft, fixedRight)

cv2.imshow('depth',depth / 16.0)