#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:13:35 2019

@author: vik748
"""
import os
import cv2
from matlab_imresize.imresize import imresize
import numpy as np
from vslam_helper import tiled_features, knn_match_and_lowe_ratio_filter
from matplotlib import pyplot as plt




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

def draw_markers(vis_orig, keypoints, color = (0, 0, 255),thickness = 2):
    if len(vis_orig.shape) == 2: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    else: vis = vis_orig
    for kp in keypoints:
        x, y = kp.pt
        cv2.drawMarker(vis, (int(x), int(y)), color,  markerSize=20, markerType = cv2.MARKER_CROSS, thickness=thickness)
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

def analyze_image_pair(image_0, image_1, settings): 
    K = settings['K']
    D = settings['D']
    TILE_KP = settings['TILE_KP']
    tiling = settings['tiling']
    zernike_detector = settings['zernike_detector']
    orb_detector = settings['orb_detector']
    sift_detector = settings['sift_detector']

    zernike_kp_0, zernike_des_0 = zernike_detector.detectAndCompute(image_0, mask=None, timing=False)
    zernike_kp_1, zernike_des_1 = zernike_detector.detectAndCompute(image_1, mask=None, timing=False)
    orb_kp_0_ut = orb_detector.detect(image_0, None)
    orb_kp_1_ut = orb_detector.detect(image_1, None)
    sift_kp_0_ut = sift_detector.detect(image_0, None)
    sift_kp_1_ut = sift_detector.detect(image_1, None)
    print ("Points before tiling supression: ",len(orb_kp_0_ut))
    
    zernike_kp_img_0 = draw_markers(image_0, zernike_kp_0, color=[255,255,0])
    zernike_kp_img_1 = draw_markers(image_1, zernike_kp_1, color=[255,255,0])
    orb_kp_img_0 = draw_markers(image_0, orb_kp_0_ut, color=[255,255,0])
    orb_kp_img_1 = draw_markers(image_1, orb_kp_1_ut, color=[255,255,0])
    sift_kp_img_0 = draw_markers(image_0, sift_kp_0_ut, color=[255,255,0])
    sift_kp_img_1 = draw_markers(image_1, sift_kp_1_ut, color=[255,255,0])
    
    
    if TILE_KP:
        orb_kp_0 = tiled_features(orb_kp_0_ut, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        orb_kp_1 = tiled_features(orb_kp_1_ut, image_1.shape, tiling[0], tiling[1], no_features= 1000)
        orb_kp_img_0 = draw_markers(orb_kp_img_0, orb_kp_0, color=[0,255,0])
        orb_kp_img_1 = draw_markers(orb_kp_img_1, orb_kp_1, color=[0,255,0])
        sift_kp_0 = tiled_features(sift_kp_0_ut, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        sift_kp_1 = tiled_features(sift_kp_1_ut, image_1.shape, tiling[0], tiling[1], no_features= 1000)
        sift_kp_img_0 = draw_markers(sift_kp_img_0, sift_kp_0, color=[0,255,0])
        sift_kp_img_1 = draw_markers(sift_kp_img_1, sift_kp_1, color=[0,255,0])
        
    else:
        orb_kp_0 = orb_kp_0_ut
        orb_kp_1 = orb_kp_1_ut
        sift_kp_0 = sift_kp_0_ut
        sift_kp_1 = sift_kp_1_ut
    
        print ("Points after tiling supression: ",len(orb_kp_0))
    
    orb_kp_0, orb_des_0 = orb_detector.compute(image_0, orb_kp_0)
    orb_kp_1, orb_des_1 = orb_detector.compute(image_1, orb_kp_1)
    sift_kp_0, sift_des_0 = sift_detector.compute(image_0, sift_kp_0)
    sift_kp_1, sift_des_1 = sift_detector.compute(image_1, sift_kp_1)
    
    fig1 = plt.figure(1); plt.clf()
    fig1, fig1_axes = plt.subplots(2,3, num=1)
    fig1.suptitle(settings['set_title'] + ' features')
    fig1_axes[0,0].axis("off"); fig1_axes[0,0].set_title("Zernike Features \n{:d} features".format(len(zernike_kp_0)))
    fig1_axes[0,0].imshow(zernike_kp_img_0)
    fig1_axes[1,0].axis("off"); fig1_axes[1,0].set_title("{:d} features".format(len(zernike_kp_1)))
    fig1_axes[1,0].imshow(zernike_kp_img_1)
    fig1_axes[0,1].axis("off"); fig1_axes[0,1].set_title("Orb Features\nBefore tiling:{:d} after tiling {:d}".format(len(orb_kp_0_ut),len(orb_kp_0)))
    fig1_axes[0,1].imshow(orb_kp_img_0)
    fig1_axes[1,1].axis("off"); fig1_axes[1,1].set_title("Before tiling:{:d} after tiling {:d}".format(len(orb_kp_1_ut),len(orb_kp_1))) 
    fig1_axes[1,1].imshow(orb_kp_img_1)
    fig1_axes[0,2].axis("off"); fig1_axes[0,2].set_title("Sift Features\nBefore tiling:{:d} after tiling {:d}".format(len(sift_kp_0_ut),len(sift_kp_0)))
    fig1_axes[0,2].imshow(sift_kp_img_0)
    fig1_axes[1,2].axis("off"); fig1_axes[1,2].set_title("Before tiling:{:d} after tiling {:d}".format(len(sift_kp_1_ut),len(sift_kp_1))) 
    fig1_axes[1,2].imshow(sift_kp_img_1)
    #fig1.subplots_adjust(0,0,1,1,0.0,0.0)
    fig1.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.1)
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
    
    zernike_valid_matches_img = draw_matches_vertical(image_0,zernike_kp_0, image_1,zernike_kp_1, zernike_matches_01, 
                                                  zernike_mask_e_12, display_invalid=True, color=(0, 255, 0))
    
    
    orb_matches_01 = knn_match_and_lowe_ratio_filter(matcher_hamming, orb_des_0, orb_des_1, threshold=0.9)
    
    orb_kp0_match_01 = np.array([orb_kp_0[mat.queryIdx].pt for mat in orb_matches_01])
    orb_kp1_match_01 = np.array([orb_kp_1[mat.trainIdx].pt for mat in orb_matches_01])
    
    orb_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp0_match_01,axis=1),K,D)
    orb_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp1_match_01,axis=1),K,D)
    
    orb_E_12, orb_mask_e_12 = cv2.findEssentialMat(orb_kp0_match_01_ud, orb_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                   method=cv2.RANSAC, prob=0.9999, threshold=0.001)
    
    print("Orb After essential: ", np.sum(orb_mask_e_12))
    
    orb_valid_matches_img = draw_matches_vertical(image_0,orb_kp_0, image_1,orb_kp_1, orb_matches_01, 
                                                  orb_mask_e_12, display_invalid=True, color=(0, 255, 0))
    
    
    sift_matches_01 = knn_match_and_lowe_ratio_filter(matcher_norm, sift_des_0, sift_des_1, threshold=0.90)
    
    sift_kp0_match_01 = np.array([sift_kp_0[mat.queryIdx].pt for mat in sift_matches_01])
    sift_kp1_match_01 = np.array([sift_kp_1[mat.trainIdx].pt for mat in sift_matches_01])
    
    sift_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(sift_kp0_match_01,axis=1),K,D)
    sift_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(sift_kp1_match_01,axis=1),K,D)
    
    sift_E_12, sift_mask_e_12 = cv2.findEssentialMat(sift_kp0_match_01_ud, sift_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                   method=cv2.RANSAC, prob=0.9999, threshold=0.001)
    
    print("sift After essential: ", np.sum(sift_mask_e_12))
    
    sift_valid_matches_img = draw_matches_vertical(image_0,sift_kp_0, image_1,sift_kp_1, sift_matches_01, 
                                                  sift_mask_e_12, display_invalid=True, color=(0, 255, 0))
    
    no_zernike_matches = np.sum(zernike_mask_e_12)
    no_orb_matches = np.sum(orb_mask_e_12)
    no_sift_matches = np.sum(sift_mask_e_12)
    
    fig2 = plt.figure(2); plt.clf()
    fig2, fig2_axes = plt.subplots(1,3, num=2)
    fig2.suptitle(settings['set_title'] + ' Feature Matching')
    fig2_axes[0].axis("off"); fig2_axes[0].set_title("Zernike Features\n{:d} matches".format(no_zernike_matches))
    fig2_axes[0].imshow(zernike_valid_matches_img)
    fig2_axes[1].axis("off"); fig2_axes[1].set_title("Orb Features\n{:d} matches".format(no_orb_matches))
    fig2_axes[1].imshow(orb_valid_matches_img)
    fig2_axes[2].axis("off"); fig2_axes[2].set_title("Sift Features\n{:d} matches".format(no_sift_matches))
    fig2_axes[2].imshow(sift_valid_matches_img)
    fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)
    
    return {'zernike_matches':no_zernike_matches, 'orb_matches':no_orb_matches, 'sift_matches':no_sift_matches}