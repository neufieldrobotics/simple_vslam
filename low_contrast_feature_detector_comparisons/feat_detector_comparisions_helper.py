#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:13:35 2019

@author: vik748
"""
import sys, os
import cv2
import numpy as np
from vslam_helper import tiled_features, knn_match_and_lowe_ratio_filter, draw_feature_tracks, draw_arrows
from matplotlib import pyplot as plt
import matplotlib as mpl
from datetime import datetime
import re
print (os.path.abspath('./external_packages/cmtpy/'))
sys.path.insert(0, os.path.abspath('./external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
from cmtpy import contrast_measurement as cm
import pandas as pd
import scipy.stats as st
import warnings

def save_fig2pdf(fig, folder=None, fname=None):
    plt._pylab_helpers.Gcf.figs.get(fig.number, None).window.showMaximized()
    plt.pause(.1)
    if fname is None:
        if fig._suptitle is None:
            fname = 'figure_{:d}'.format(fig.number)
        else:
            ttl = fig._suptitle.get_text()
            ttl = ttl.replace('$','').replace('\n','_').replace(' ','_')
            fname = re.sub(r"\_\_+", "_", ttl) 
    if folder:
        plt.savefig(os.path.join(folder, fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf'),format='pdf', dpi=1200,  orientation='landscape', papertype='letter')
    else:
        plt.savefig(fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf',format='pdf', dpi=1200,  orientation='landscape', papertype='letter')

    plt._pylab_helpers.Gcf.figs.get(fig.number, None).window.showNormal()
    
def save_fig2png(fig, size=[8, 6.7], folder=None, fname=None):    
    if size is None:
        fig.set_size_inches([8, 6.7])
    else:
        fig.set_size_inches(size)
    plt.pause(.1)
    if fname is None:
        if fig._suptitle is None:
            fname = 'figure_{:d}'.format(fig.number)
        else:
            ttl = fig._suptitle.get_text()
            ttl = ttl.replace('$','').replace('\n','_').replace(' ','_')
            fname = re.sub(r"\_\_+", "_", ttl) 
    if folder:
        plt.savefig(os.path.join(folder, fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf'),format='pdf', dpi=1200,  orientation='landscape', papertype='letter')
    else:
        plt.savefig(fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.png',format='png', dpi=300)

    

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

def draw_markers(vis_orig, keypoints, color = (0, 0, 255),thickness = 2, 
                 markerType=cv2.MARKER_CROSS, markerSize=20):
    if len(vis_orig.shape) == 2: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    else: vis = vis_orig
    if isinstance(color,list) and len(color) != 3:
        assert len(keypoints) == len(color)
        for kp,cl in zip(keypoints,color):
            x, y = kp.pt
            cv2.drawMarker(vis, (int(x), int(y)), cl,  
                           markerSize=markerSize, markerType = markerType, thickness=thickness)
    else:    
        for kp in keypoints:
            x, y = kp.pt
            cv2.drawMarker(vis, (int(x), int(y)), color,  
                           markerSize=markerSize, markerType = markerType, thickness=thickness)
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

def analyze_image_pair_zer_orb_sift(image_0, image_1, settings, plotMatches=True): 
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
    #print ("Points before tiling supression: ",len(orb_kp_0_ut))
        
    
    if TILE_KP:
        orb_kp_0 = tiled_features(orb_kp_0_ut, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        orb_kp_1 = tiled_features(orb_kp_1_ut, image_1.shape, tiling[0], tiling[1], no_features= 1000)
        sift_kp_0 = tiled_features(sift_kp_0_ut, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        sift_kp_1 = tiled_features(sift_kp_1_ut, image_1.shape, tiling[0], tiling[1], no_features= 1000)
        
    else:
        orb_kp_0 = orb_kp_0_ut
        orb_kp_1 = orb_kp_1_ut
        sift_kp_0 = sift_kp_0_ut
        sift_kp_1 = sift_kp_1_ut
    
        #print ("Points after tiling supression: ",len(orb_kp_0))
    
    
    if plotMatches:
        zernike_kp_img_0 = draw_markers(image_0, zernike_kp_0, color=[255,255,0])
        zernike_kp_img_1 = draw_markers(image_1, zernike_kp_1, color=[255,255,0])
        orb_kp_img_0 = draw_markers(image_0, orb_kp_0_ut, color=[255,255,0])
        orb_kp_img_1 = draw_markers(image_1, orb_kp_1_ut, color=[255,255,0])
        sift_kp_img_0 = draw_markers(image_0, sift_kp_0_ut, color=[255,255,0])
        sift_kp_img_1 = draw_markers(image_1, sift_kp_1_ut, color=[255,255,0])

        if TILE_KP:
            orb_kp_img_0 = draw_markers(orb_kp_img_0, orb_kp_0, color=[0,255,0])
            orb_kp_img_1 = draw_markers(orb_kp_img_1, orb_kp_1, color=[0,255,0])
            sift_kp_img_0 = draw_markers(sift_kp_img_0, sift_kp_0, color=[0,255,0])
            sift_kp_img_1 = draw_markers(sift_kp_img_1, sift_kp_1, color=[0,255,0])                
        
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
        plt.draw(); plt.show(block=False)

    orb_kp_0, orb_des_0 = orb_detector.compute(image_0, orb_kp_0)
    orb_kp_1, orb_des_1 = orb_detector.compute(image_1, orb_kp_1)
    sift_kp_0, sift_des_0 = sift_detector.compute(image_0, sift_kp_0)
    sift_kp_1, sift_des_1 = sift_detector.compute(image_1, sift_kp_1)
    
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
    
    #print("Zernike After essential: ", np.sum(zernike_mask_e_12))
    
    
    
    orb_matches_01 = knn_match_and_lowe_ratio_filter(matcher_hamming, orb_des_0, orb_des_1, threshold=0.9)
    
    orb_kp0_match_01 = np.array([orb_kp_0[mat.queryIdx].pt for mat in orb_matches_01])
    orb_kp1_match_01 = np.array([orb_kp_1[mat.trainIdx].pt for mat in orb_matches_01])
    
    orb_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp0_match_01,axis=1),K,D)
    orb_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp1_match_01,axis=1),K,D)
    
    orb_E_12, orb_mask_e_12 = cv2.findEssentialMat(orb_kp0_match_01_ud, orb_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                   method=cv2.RANSAC, prob=0.9999, threshold=0.001)
    
    #print("Orb After essential: ", np.sum(orb_mask_e_12))
    
    sift_matches_01 = knn_match_and_lowe_ratio_filter(matcher_norm, sift_des_0, sift_des_1, threshold=0.90)
    
    sift_kp0_match_01 = np.array([sift_kp_0[mat.queryIdx].pt for mat in sift_matches_01])
    sift_kp1_match_01 = np.array([sift_kp_1[mat.trainIdx].pt for mat in sift_matches_01])
    
    sift_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(sift_kp0_match_01,axis=1),K,D)
    sift_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(sift_kp1_match_01,axis=1),K,D)
    
    sift_E_12, sift_mask_e_12 = cv2.findEssentialMat(sift_kp0_match_01_ud, sift_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                   method=cv2.RANSAC, prob=0.9999, threshold=0.001)
    
    #print("sift After essential: ", np.sum(sift_mask_e_12))
    
    no_zernike_matches = np.sum(zernike_mask_e_12)
    no_orb_matches = np.sum(orb_mask_e_12)
    no_sift_matches = np.sum(sift_mask_e_12)
    
    if plotMatches:
    
        zernike_valid_matches_img = draw_matches_vertical(image_0,zernike_kp_0, image_1,zernike_kp_1, zernike_matches_01, 
                                                          zernike_mask_e_12, display_invalid=True, color=(0, 255, 0))
    
        orb_valid_matches_img = draw_matches_vertical(image_0,orb_kp_0, image_1,orb_kp_1, orb_matches_01, 
                                                      orb_mask_e_12, display_invalid=True, color=(0, 255, 0))
        
        sift_valid_matches_img = draw_matches_vertical(image_0,sift_kp_0, image_1,sift_kp_1, sift_matches_01, 
                                                       sift_mask_e_12, display_invalid=True, color=(0, 255, 0))
            
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
        plt.draw(); plt.show(block=False)
        input("Enter to continue")

    return {'zernike_matches':no_zernike_matches, 'orb_matches':no_orb_matches, 'sift_matches':no_sift_matches}

def analyze_image_pair_zer_orb_orbhc(image_0, image_1, settings, plotMatches=True): 
    K = settings['K']
    D = settings['D']
    TILE_KP = settings['TILE_KP']
    tiling = settings['tiling']
    zernike_detector = settings['zernike_detector']
    orb_detector = settings['orb_detector']

    zernike_kp_0, zernike_des_0 = zernike_detector.detectAndCompute(image_0, mask=None, timing=False)
    zernike_kp_1, zernike_des_1 = zernike_detector.detectAndCompute(image_1, mask=None, timing=False)
    orb_kp_0_ut = orb_detector.detect(image_0, None)
    orb_kp_1_ut = orb_detector.detect(image_1, None)
    # Use Harris corners from zernike for ORB
    orbhc_kp_0 = zernike_kp_0.copy()
    orbhc_kp_1 = zernike_kp_1.copy()
    orbhc_kp_0, orbhc_des_0 = orb_detector.compute(image_0, orbhc_kp_0)
    orbhc_kp_1, orbhc_des_1 = orb_detector.compute(image_1, orbhc_kp_1)
    
    if TILE_KP:
        orb_kp_0 = tiled_features(orb_kp_0_ut, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        orb_kp_1 = tiled_features(orb_kp_1_ut, image_1.shape, tiling[0], tiling[1], no_features= 1000)
        
    else:
        orb_kp_0 = orb_kp_0_ut
        orb_kp_1 = orb_kp_1_ut
        
    if plotMatches:
        zernike_kp_img_0 = draw_markers(image_0, zernike_kp_0, color=[255,255,0])
        zernike_kp_img_1 = draw_markers(image_1, zernike_kp_1, color=[255,255,0])
        orb_kp_img_0 = draw_markers(image_0, orb_kp_0_ut, color=[255,255,0])
        orb_kp_img_1 = draw_markers(image_1, orb_kp_1_ut, color=[255,255,0])
        orbhc_kp_img_0 = draw_markers(image_0, orbhc_kp_0, color=[255,255,0])
        orbhc_kp_img_1 = draw_markers(image_1, orbhc_kp_1, color=[255,255,0])
        

        if TILE_KP:
            orb_kp_img_0 = draw_markers(orb_kp_img_0, orb_kp_0, color=[0,255,0])
            orb_kp_img_1 = draw_markers(orb_kp_img_1, orb_kp_1, color=[0,255,0])
        
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
        fig1_axes[0,2].axis("off"); fig1_axes[0,2].set_title("ORB Harris Corner Features\n{:d} features".format(len(orbhc_kp_0)))
        fig1_axes[0,2].imshow(orbhc_kp_img_0)
        fig1_axes[1,2].axis("off"); fig1_axes[1,2].set_title("{:d} features".format(len(orbhc_kp_1))) 
        fig1_axes[1,2].imshow(orbhc_kp_img_1)
        #fig1.subplots_adjust(0,0,1,1,0.0,0.0)
        fig1.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.1)
        plt.draw(); plt.show(block=False)

    orb_kp_0, orb_des_0 = orb_detector.compute(image_0, orb_kp_0)
    orb_kp_1, orb_des_1 = orb_detector.compute(image_1, orb_kp_1)
    
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

    
    orb_matches_01 = knn_match_and_lowe_ratio_filter(matcher_hamming, orb_des_0, orb_des_1, threshold=0.9)
    
    orb_kp0_match_01 = np.array([orb_kp_0[mat.queryIdx].pt for mat in orb_matches_01])
    orb_kp1_match_01 = np.array([orb_kp_1[mat.trainIdx].pt for mat in orb_matches_01])
    
    orb_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp0_match_01,axis=1),K,D)
    orb_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(orb_kp1_match_01,axis=1),K,D)
    
    orb_E_12, orb_mask_e_12 = cv2.findEssentialMat(orb_kp0_match_01_ud, orb_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                   method=cv2.RANSAC, prob=0.9999, threshold=0.001)
       
    
    orbhc_matches_01 = knn_match_and_lowe_ratio_filter(matcher_hamming, orbhc_des_0, orbhc_des_1, threshold=0.9)
    
    orbhc_kp0_match_01 = np.array([orbhc_kp_0[mat.queryIdx].pt for mat in orbhc_matches_01])
    orbhc_kp1_match_01 = np.array([orbhc_kp_1[mat.trainIdx].pt for mat in orbhc_matches_01])
    
    orbhc_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(orbhc_kp0_match_01,axis=1),K,D)
    orbhc_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(orbhc_kp1_match_01,axis=1),K,D)
    
    orbhc_E_12, orbhc_mask_e_12 = cv2.findEssentialMat(orbhc_kp0_match_01_ud, orbhc_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                       method=cv2.RANSAC, prob=0.9999, threshold=0.001)
    
    
    no_zernike_matches = np.sum(zernike_mask_e_12)
    no_orb_matches = np.sum(orb_mask_e_12)
    no_orbhc_matches = np.sum(orbhc_mask_e_12)
    
    if plotMatches:
    
        zernike_valid_matches_img = draw_matches_vertical(image_0,zernike_kp_0, image_1,zernike_kp_1, zernike_matches_01, 
                                                          zernike_mask_e_12, display_invalid=True, color=(0, 255, 0))
    
        orb_valid_matches_img = draw_matches_vertical(image_0,orb_kp_0, image_1,orb_kp_1, orb_matches_01, 
                                                      orb_mask_e_12, display_invalid=True, color=(0, 255, 0))
        
        orbhc_valid_matches_img = draw_matches_vertical(image_0,orbhc_kp_0, image_1,orbhc_kp_1, orbhc_matches_01, 
                                                        orbhc_mask_e_12, display_invalid=True, color=(0, 255, 0))
            
        fig2 = plt.figure(2); plt.clf()
        fig2, fig2_axes = plt.subplots(1,3, num=2)
        fig2.suptitle(settings['set_title'] + ' Feature Matching')
        fig2_axes[0].axis("off"); fig2_axes[0].set_title("Zernike Features\n{:d} matches".format(no_zernike_matches))
        fig2_axes[0].imshow(zernike_valid_matches_img)
        fig2_axes[1].axis("off"); fig2_axes[1].set_title("Orb Features\n{:d} matches".format(no_orb_matches))
        fig2_axes[1].imshow(orb_valid_matches_img)
        fig2_axes[2].axis("off"); fig2_axes[2].set_title("Orb Harris Corner Features\n{:d} matches".format(no_orbhc_matches))
        fig2_axes[2].imshow(orbhc_valid_matches_img)
        fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)
        plt.draw(); plt.show(block=False)
        input("Enter to continue")

    return {'zernike_matches':no_zernike_matches, 'orb_matches':no_orb_matches, 'orbhc_matches':no_orbhc_matches}

def analyze_image_pair_zer_surf_orbsf(image_0, image_1, settings, plotMatches=True): 
    K = settings['K']
    D = settings['D']
    TILE_KP = settings['TILE_KP']
    tiling = settings['tiling']
    zernike_detector = settings['zernike_detector']
    orb_detector = settings['orb_detector']
    surf_detector = settings['surf_detector']

    zernike_kp_0, zernike_des_0 = zernike_detector.detectAndCompute(image_0, mask=None, timing=False)
    zernike_kp_1, zernike_des_1 = zernike_detector.detectAndCompute(image_1, mask=None, timing=False)
    surf_kp_0_ut = surf_detector.detect(image_0, mask=None)
    surf_kp_1_ut = surf_detector.detect(image_1, mask=None)
    # Use Harris corners from zernike for ORB
    
    if TILE_KP:
        surf_kp_0 = tiled_features(surf_kp_0_ut, image_0.shape, tiling[0], tiling[1], no_features= 1000)
        surf_kp_1 = tiled_features(surf_kp_1_ut, image_1.shape, tiling[0], tiling[1], no_features= 1000)
        
    else:
        surf_kp_0 = surf_kp_0_ut
        surf_kp_1 = surf_kp_1_ut
        
    orbsf_kp_0 = surf_kp_0.copy()   
    orbsf_kp_1 = surf_kp_1.copy()  
    orbsf_kp_0, orbsf_des_0 = orb_detector.compute(image_0, orbsf_kp_0)
    orbsf_kp_1, orbsf_des_1 = orb_detector.compute(image_1, orbsf_kp_1)
    surf_kp_0, surf_des_0 = surf_detector.compute(image_0, surf_kp_0)
    surf_kp_1, surf_des_1 = surf_detector.compute(image_1, surf_kp_1)
    

    if plotMatches:
        zernike_kp_img_0 = draw_markers(image_0, zernike_kp_0, color=[255,255,0])
        zernike_kp_img_1 = draw_markers(image_1, zernike_kp_1, color=[255,255,0])
        surf_kp_img_0 = draw_markers(image_0, surf_kp_0_ut, color=[255,255,0])
        surf_kp_img_1 = draw_markers(image_1, surf_kp_1_ut, color=[255,255,0])
        orbsf_kp_img_0 = draw_markers(image_0, surf_kp_0_ut, color=[255,255,0])
        orbsf_kp_img_1 = draw_markers(image_1, surf_kp_1_ut, color=[255,255,0])
        

        if TILE_KP:
            surf_kp_img_0 = draw_markers(surf_kp_img_0, surf_kp_0, color=[0,255,0])
            surf_kp_img_1 = draw_markers(surf_kp_img_1, surf_kp_1, color=[0,255,0])
            orbsf_kp_img_0 = draw_markers(orbsf_kp_img_0, orbsf_kp_0, color=[0,255,0])
            orbsf_kp_img_1 = draw_markers(orbsf_kp_img_1, orbsf_kp_1, color=[0,255,0])
        
        fig1 = plt.figure(1); plt.clf()
        fig1, fig1_axes = plt.subplots(2,3, num=1)
        fig1.suptitle(settings['set_title'] + ' features')
        fig1_axes[0,0].axis("off"); fig1_axes[0,0].set_title("Zernike Features \n{:d} features".format(len(zernike_kp_0)))
        fig1_axes[0,0].imshow(zernike_kp_img_0)
        fig1_axes[1,0].axis("off"); fig1_axes[1,0].set_title("{:d} features".format(len(zernike_kp_1)))
        fig1_axes[1,0].imshow(zernike_kp_img_1)
        fig1_axes[0,1].axis("off"); fig1_axes[0,1].set_title("SURF Features\nBefore tiling:{:d} after tiling {:d}".format(len(surf_kp_0_ut),len(surf_kp_0)))
        fig1_axes[0,1].imshow(surf_kp_img_0)
        fig1_axes[1,1].axis("off"); fig1_axes[1,1].set_title("Before tiling:{:d} after tiling {:d}".format(len(surf_kp_1_ut),len(surf_kp_1))) 
        fig1_axes[1,1].imshow(surf_kp_img_1)
        fig1_axes[0,2].axis("off"); fig1_axes[0,2].set_title("ORB SURF Corners Features\n{:d} features".format(len(orbsf_kp_0)))
        fig1_axes[0,2].imshow(orbsf_kp_img_0)
        fig1_axes[1,2].axis("off"); fig1_axes[1,2].set_title("{:d} features".format(len(orbsf_kp_1))) 
        fig1_axes[1,2].imshow(orbsf_kp_img_1)
        #fig1.subplots_adjust(0,0,1,1,0.0,0.0)
        fig1.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.1)
        plt.draw(); plt.show(block=False)

    
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

    
    surf_matches_01 = knn_match_and_lowe_ratio_filter(matcher_norm, surf_des_0, surf_des_1, threshold=0.9)
    
    surf_kp0_match_01 = np.array([surf_kp_0[mat.queryIdx].pt for mat in surf_matches_01])
    surf_kp1_match_01 = np.array([surf_kp_1[mat.trainIdx].pt for mat in surf_matches_01])
    
    surf_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(surf_kp0_match_01,axis=1),K,D)
    surf_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(surf_kp1_match_01,axis=1),K,D)
    
    surf_E_12, surf_mask_e_12 = cv2.findEssentialMat(surf_kp0_match_01_ud, surf_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                   method=cv2.RANSAC, prob=0.9999, threshold=0.001)
       
    
    orbsf_matches_01 = knn_match_and_lowe_ratio_filter(matcher_hamming, orbsf_des_0, orbsf_des_1, threshold=0.9)
    
    orbsf_kp0_match_01 = np.array([orbsf_kp_0[mat.queryIdx].pt for mat in orbsf_matches_01])
    orbsf_kp1_match_01 = np.array([orbsf_kp_1[mat.trainIdx].pt for mat in orbsf_matches_01])
    
    orbsf_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(orbsf_kp0_match_01,axis=1),K,D)
    orbsf_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(orbsf_kp1_match_01,axis=1),K,D)
    
    orbsf_E_12, orbsf_mask_e_12 = cv2.findEssentialMat(orbsf_kp0_match_01_ud, orbsf_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                                       method=cv2.RANSAC, prob=0.9999, threshold=0.001)
    
    
    no_zernike_matches = np.sum(zernike_mask_e_12)
    no_surf_matches = np.sum(surf_mask_e_12)
    no_orbsf_matches = np.sum(orbsf_mask_e_12)
    
    if plotMatches:
    
        zernike_valid_matches_img = draw_matches_vertical(image_0,zernike_kp_0, image_1,zernike_kp_1, zernike_matches_01, 
                                                          zernike_mask_e_12, display_invalid=True, color=(0, 255, 0))
    
        surf_valid_matches_img = draw_matches_vertical(image_0,surf_kp_0, image_1,surf_kp_1, surf_matches_01, 
                                                      surf_mask_e_12, display_invalid=True, color=(0, 255, 0))
        
        orbsf_valid_matches_img = draw_matches_vertical(image_0,orbsf_kp_0, image_1,orbsf_kp_1, orbsf_matches_01, 
                                                        orbsf_mask_e_12, display_invalid=True, color=(0, 255, 0))
            
        fig2 = plt.figure(2); plt.clf()
        fig2, fig2_axes = plt.subplots(1,3, num=2)
        fig2.suptitle(settings['set_title'] + ' Feature Matching')
        fig2_axes[0].axis("off"); fig2_axes[0].set_title("Zernike Features\n{:d} matches".format(no_zernike_matches))
        fig2_axes[0].imshow(zernike_valid_matches_img)
        fig2_axes[1].axis("off"); fig2_axes[1].set_title("surf Features\n{:d} matches".format(no_surf_matches))
        fig2_axes[1].imshow(surf_valid_matches_img)
        fig2_axes[2].axis("off"); fig2_axes[2].set_title("Orb SURF Corner Features\n{:d} matches".format(no_orbsf_matches))
        fig2_axes[2].imshow(orbsf_valid_matches_img)
        fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)
        plt.draw(); plt.show(block=False)
        input("Enter to continue")

    return {'zernike_matches':no_zernike_matches, 'surf_matches':no_surf_matches, 'orbsf_matches':no_orbsf_matches}

def analyze_image_pair(image_0, image_1, settings, plotMatches=True, saveFig=False): 
    #K = settings['K']
    #D = settings['D']
    
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
        fig1, fig1_axes = plt.subplots(2,1,num=1)
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
    return result

def generate_contrast_images(img, mask=None, contrast_adj_factors=np.arange(0,-1.1,-.1)):
    '''
    Given an image and list of contrsat_adj_factors returns list of images and contrat measurements
    '''
    ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, lam=5, adjustment_factor=-1.0, stretch_factor=-1.0,
                                  min_stretch_bits=3, downsample_for_kde=True,debug=False, plot_histograms=False)
    v_k, a_k = ace_obj.compute_vk_and_ak(img)

    warped_images = np.empty(contrast_adj_factors.shape,dtype=object)
    contrast_measurements = []
    
    contrast_estimators = {'global_contrast_factor': lambda gr: cm.compute_global_contrast_factor(gr),
                           'rms_contrast': lambda gr: cm.compute_rms_contrast(gr,mask=mask,debug=False),
                           'local_box_filt': lambda gr: cm.compute_box_filt_contrast(gr,mask=mask, kernel_size=17, debug=False),
                           'local_gaussian_filt': lambda gr: cm.compute_gaussian_filt_contrast(gr,mask=mask, sigma=5.0, debug=False),
                           'local_bilateral_filt': lambda gr: cm.compute_bilateral_filt_contrast(gr,mask=mask, sigmaSpace=5.0, sigmaColor=0.05, debug=False)}
        
    for i,adj in enumerate(contrast_adj_factors):
        if adj==0:
            warped_images[i] = img
        else:
            outputs = ace_obj.compute_bk_and_dk(v_k, a_k, adjustment_factor=adj, stretch_factor=adj)
            warped_images[i], Tx = ace_obj.transform_image(*outputs, img)
        #cm_dict = {nm:ce(warped_images[i]) for nm, ce in contrast_estimators.items()}
        
        cm_dict = {}
        for nm,ce in contrast_estimators.items():
            contrast, contrast_masked = ce(warped_images[i])
            cm_dict.update({nm:contrast, nm+"_masked":contrast_masked})
        #print("Contrast adj: {:.2f} contrast estimaes: {}".format(adj, cm_dict))
        contrast_measurements.append(cm_dict)
        
    return warped_images, contrast_measurements

def read_grimage(img_name, resize_scale = None, normalize=False, image_depth=8):
    '''
    Read image from file, convert to grayscale and resize if required
    Parameters
    ----------
    img_name : String
        Filename
    resize_scale : float, optional
        float scale factor for image < 1 downsamples and > 1 upsamples. The default is None.
    normalize : bool, optional
        Return normalized float image. The default is False.
    image_depth : int, optional
        Bit depth of image being read. The default is 8.

    Raises
    ------
    FileNotFoundError
        Raisees FileNotFound Error if unable to read image

    Returns
    -------
    gr : MxN uint8 or flat32 numpy array
        grayscale image
    '''
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img is None:
        #raise FileNotFoundError ("Could not read image from: {}".format(img_name))
        warnings.warn("Could not read image from: {}".format(img_name))
        return None
    gr_full = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if not resize_scale is None:
        gr = cv2.resize(gr_full, (0,0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
    else:
        gr = gr_full

    if normalize:
        levels = 2 ** image_depth - 1
        gr = np.divide(gr, levels, dtype=np.float32)

    return gr

def process_image_contrasts(img_name, contrast_adj_factors, mask_folder, ctrst_img_output_folder, base_settings):
    print("Processing image: {}".format(img_name))
    img_name_base, img_name_ext = os.path.splitext(os.path.basename(img_name))
    img = read_grimage(img_name)
    if mask_folder is not None:
        mask_name = os.path.join(mask_folder, img_name_base+'_mask'+'.png')
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE).astype(bool)
    else:
        mask = None
    
    img_df = pd.DataFrame(columns = ['set_title','image_name', 'contrast_adj_factor',
                                     'global_contrast_factor', 'rms_contrast', 'local_box_filt','local_gaussian_filt', 'local_bilateral_filt',
                                     'global_contrast_factor_masked', 'rms_contrast_masked', 'local_box_filt_masked','local_gaussian_filt_masked', 'local_bilateral_filt_masked'])
        
    contrast_imgs, contrast_meas = generate_contrast_images(img, mask=mask, contrast_adj_factors=contrast_adj_factors)
    for c_img, c_meas, adj in zip(contrast_imgs, contrast_meas, contrast_adj_factors):
        out_img_name = os.path.join(ctrst_img_output_folder, img_name_base+"_ctrst_adj_{:.2f}.png".format(adj) )
        print("\t"+out_img_name)
        cv2.imwrite(out_img_name, c_img)
        img_df=img_df.append({'image_name':img_name_base, 'contrast_adj_factor':adj, **c_meas, **base_settings},
                             ignore_index=True)
    return img_df

def preprocessed_image_contrasts(img_name, contrast_adj_factors, contrast_img_folder, contrast_img_df):
    img_base, img_ext = os.path.splitext(os.path.basename(img_name))
    raw_sets_folder = os.path.basename(os.path.dirname(img_name))
    
    contrast_imgs = []
    contrast_meas = []
    for ctrst_adj_fact in contrast_adj_factors:
        ctrst_adj_img_basename = "{}_ctrst_adj_{:.2f}{}".format(img_base, ctrst_adj_fact,img_ext)
        ctrst_adj_img_name = os.path.join(contrast_img_folder, ctrst_adj_img_basename)
        contrast_imgs.append( read_grimage(ctrst_adj_img_name) )
        contrast_meas.append( contrast_img_df.loc[(raw_sets_folder,img_base, ctrst_adj_fact), :].to_dict() )
        
    return contrast_imgs, contrast_meas

def normalize_and_applycolormap(values, values_max = None, colormap = cv2.COLORMAP_RAINBOW):
    '''
    Given an array of values, normalize to max value and apply colormap
    '''
    if values_max is not None:
        intensities = np.round(values/values_max * 255).astype('uint8')
    else:
        intensities = np.round(values/values.max() * 255).astype('uint8')
    colors = cv2.applyColorMap(intensities, colormap)
    color_int_tuples = [tuple(map(int, cl)) for cl in colors[:,0,:]]
    return color_int_tuples

def analyze_descriptor_distance_image_pair(image_0, settings, mask=None,
                                           plotMatches=True, saveFig=False):
    '''
    Compute distances to the 2nd clostest features and plot histogram + kde
    '''
    TILE_KP = settings['TILE_KP']
    tiling = settings['tiling']
    detector = settings['detector']
    descriptor = settings['descriptor']
    max_prob = settings.get('max_prob')
    provided_kps = settings.get('provided_keypoints')

    det_name = detector.getDefaultName().replace('Feature2D.','')
    if det_name == 'Feature2D':
        det_name = type(detector).__name__.replace('xfeatures2d_','')
    des_name = descriptor.getDefaultName().replace('Feature2D.','')
    if des_name == 'Feature2D':
        des_name = type(descriptor).__name__.replace('xfeatures2d_','')
    
    # CASE WHEN FEATURES are PROVIDED
    if provided_kps is not None:
        det_name += '_provided'
        kp_0, des_0 = descriptor.compute(image_0, provided_kps)

    else:
        NO_OF_FEATURES = settings['NO_OF_FEATURES']

        if isinstance(descriptor, cv2.SparsePyrLKOpticalFlow): des_name='SparsePyrLKOpticalFlow'
            
        if detector == descriptor and not TILE_KP:
            kp_0, des_0 = detector.detectAndCompute(image_0, mask=mask)
            #kp_1, des_1 = detector.detectAndCompute(image_1, mask=None)
        else:
            detector_kp_0 = detector.detect(image_0, mask=mask)
        
            if TILE_KP:
                try:
                    kp_0 = tiled_features(detector_kp_0, image_0.shape, tiling[0], tiling[1], no_features= NO_OF_FEATURES)
                except:
                    print("Error with tiling, probably not enough features: {:d}".format(len(detector_kp_0)))
                    print("Continuing without tiling")            
                    kp_0 = detector_kp_0
            else:
                kp_0 = detector_kp_0
        
            if not isinstance(descriptor, cv2.SparsePyrLKOpticalFlow):
                kp_0, des_0 = descriptor.compute(image_0, kp_0)
       
    if len(kp_0) == 0:
        return {'detector':det_name, 'descriptor':des_name,
                'second_dist_kde': None, 'kde_x':None, 'img0_no_features': len(kp_0)}

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
    
    second_dist_max = settings.get('max_second_dist',second_distances.max())
    marker_colors = normalize_and_applycolormap(second_distances, values_max=second_dist_max, colormap = cv2.COLORMAP_RAINBOW)
    
    #no_matches = np.sum(mask_e_12)
    

    x = np.linspace(0,second_dist_max,51)
    sec_dist_kde_full = st.gaussian_kde(second_distances,bw_method='silverman')
    sec_dist_kde = sec_dist_kde_full(x)
        
    if plotMatches:        
        full_map_BGR = cv2.applyColorMap(np.linspace(0,255,256).astype(np.uint8),cv2.COLORMAP_RAINBOW)
        full_map_RGB = cv2.cvtColor(full_map_BGR, cv2.COLOR_BGR2RGB)
        full_map = full_map_RGB[:,0,:] /255
        cmap= mpl.colors.ListedColormap(full_map)
        norm= mpl.colors.Normalize(vmin=0,vmax=second_dist_max)
        print("Second dist max ", second_dist_max)
        
        det_des_string = "Det: {} Des: {}".format(det_name, des_name)
        kp_img_0 = image_0
        
        if TILE_KP:
            feat_string_0 = "{}\nBefore tiling:{:d} after tiling {:d}".format(det_des_string, len(detector_kp_0), len(kp_0))
    #        kp_img_0 = draw_markers(kp_img_0, detector_kp_0, color=marker_colors, markerSize=1)
            
        else:            
            feat_string_0 = "{}\n{:d} features".format(det_des_string, len(kp_0))
    
        kp_img_0 = draw_markers(kp_img_0, kp_0, color=marker_colors)
    
        #fig1 = plt.figure(1); plt.clf()
        fig1, fig1_axes = plt.subplots(2,1, figsize= [9.28, 9.58])
        fig1.subplots_adjust(top=0.91)
        fig1.suptitle(settings['set_title'] + ' features')
        fig1_axes[0].axis("off"); fig1_axes[0].set_title(feat_string_0)
        fig1_axes[0].imshow(cv2.cvtColor(kp_img_0, cv2.COLOR_BGR2RGB))        
        #cb = matplotlib.colorbar.ColorbarBase(fig1_axes[2], orientation='horizontal', 
        #                           cmap=cmap, norm=norm, fraction=1)
        
        adjust_lower_axes_up = .06
        bb_0 = fig1_axes[0].get_position()
        left0, bottom0, width0, height0 = bb_0.x0, bb_0.y0, bb_0.width, bb_0.height
        bb_1 = fig1_axes[1].get_position()
        left1, bottom1, width1, height1 = bb_1.x0, bb_1.y0+adjust_lower_axes_up, bb_1.width, bb_1.height
        fig1_axes[1].set_position([left0, bottom1, width0, height1])
        
        fig1_axes[1].hist(second_distances, bins=x, color='blue', density=True, alpha=0.4, label='Raw')
        fig1_axes[1].fill_between(x, sec_dist_kde, color='red',alpha=0.4)        
        fig1_axes[1].set_xlim((0,second_dist_max))
        fig1_axes[1].set_xticklabels([])
        fig1_axes[1].set_ylim([0,max_prob])
        fig1_axes[1].set_ylabel('Prob. density')
    
        #im=mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        #fig1.colorbar(im, ax=fig1_axes[1], cmap=cmap, orientation='horizontal', norm=norm)

        cb_ax = fig1.add_axes([left0, 0.08+adjust_lower_axes_up, width0, 0.025])
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap,
                                       norm=norm, orientation='horizontal')
        cb_ax.set_xlabel('Distance to second closest matches')
        
        fig_ttl = fig1._suptitle.get_text() + '_' + fig1_axes[0].get_title()
        fig_ttl = fig_ttl.replace('$','').replace('\n','_').replace(' ','_')
        fig_fname = re.sub(r"\_\_+", "_", fig_ttl)

        if saveFig:
            save_fig2png(fig1, fname= fig_fname, size = [9.28, 9.58])
    
    result = {'detector':det_name, 'descriptor':des_name,
              'second_dist_kde': sec_dist_kde, 'kde_x':x, 'img0_no_features': len(kp_0), 
              'keypoints':kp_0}
    return result

def adjust_lower_axes_to_image(axes, adj1_up = 0.04, adj2_up = 0.005):
    '''
    In a figure with 3 vertical axes, make axes 1 and 2 as wide as image in 0
    Also adjust them vertically to make them tight
    Also return an axes for colorbar touching lower x axis
    '''
    bb_0 = axes[0].get_position()
    left0, bottom0, width0, height0 = bb_0.x0, bb_0.y0, bb_0.width, bb_0.height

    bb_1 = axes[1].get_position()
    left1, bottom1, width1, height1 = bb_1.x0, bb_1.y0, bb_1.width, bb_1.height    
    axes[1].set_position([left0, bottom1+adj1_up, width0, height1])
        
    bb_2 = axes[2].get_position()
    left2, bottom2, width2, height2 = bb_2.x0, bb_2.y0 + adj2_up, bb_2.width, bb_2.height
    axes[2].set_position([left0, bottom2+ adj1_up + adj2_up, width0, height1])
    
    cb_ax = axes[0].figure.add_axes([left0, 0.08 + adj1_up + adj2_up, width0, 0.025])
    return cb_ax


def analyze_descriptor_distance_image_pair_with_eig(image_0, settings, mask=None,
                                                    plotMatches=True, saveFig=False):
    '''
    Compute distances to the 2nd clostest features and plot histogram + kde
    '''
    TILE_KP = settings['TILE_KP']
    tiling = settings['tiling']
    detector = settings['detector']
    descriptor = settings['descriptor']
    max_prob = settings.get('max_prob')
    provided_kps = settings.get('provided_keypoints')
    kp_eig_vals_old = settings.get('kp_eigen_values')
    provided_Ft = settings.get('provided_Ft')
    plot_eig_movement = settings.get('plot_eig_movement',False)
    eig_data_lims = settings.get('eig_plot_lims')

    det_name = detector.getDefaultName().replace('Feature2D.','')
    if det_name == 'Feature2D':
        det_name = type(detector).__name__.replace('xfeatures2d_','')
    des_name = descriptor.getDefaultName().replace('Feature2D.','')
    if des_name == 'Feature2D':
        des_name = type(descriptor).__name__.replace('xfeatures2d_','')
    
    # CASE WHEN FEATURES are PROVIDED
    if provided_Ft is not None:
        print("Using Ft")
        det_name += '_provided'
        kp_0, des_0, Ft, eig_vals = descriptor.computeFromFt(image_0, provided_Ft, computeEigVals=True)
        kp_eig_vals = eig_vals

    elif provided_kps is not None:
        print("Using Provided keypoints")
        det_name += '_provided'
        kp_0, des_0 = descriptor.compute(image_0, provided_kps)
            
    else:
        NO_OF_FEATURES = settings['NO_OF_FEATURES']

        if isinstance(descriptor, cv2.SparsePyrLKOpticalFlow): des_name='SparsePyrLKOpticalFlow'
            
        if detector == descriptor and not TILE_KP:
            kp_0, des_0 = detector.detectAndCompute(image_0, mask=mask)
            #kp_1, des_1 = detector.detectAndCompute(image_1, mask=None)
        else:
            detector_kp_0 = detector.detect(image_0, mask=mask)
        
            if TILE_KP:
                try:
                    kp_0 = tiled_features(detector_kp_0, image_0.shape, tiling[0], tiling[1], no_features= NO_OF_FEATURES)
                except:
                    print("Error with tiling, probably not enough features: {:d}".format(len(detector_kp_0)))
                    print("Continuing without tiling")            
                    kp_0 = detector_kp_0
            else:
                kp_0 = detector_kp_0
        
            if not isinstance(descriptor, cv2.SparsePyrLKOpticalFlow):
                kp_0, des_0 = descriptor.compute(image_0, kp_0)
       
    if len(kp_0) == 0:
        return {'detector':det_name, 'descriptor':des_name,
                'second_dist_kde': None, 'kde_x':None, 'img0_no_features': len(kp_0)}

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
    
    second_dist_max = settings.get('max_second_dist',second_distances.max())
    marker_colors = normalize_and_applycolormap(second_distances, values_max=second_dist_max, colormap = cv2.COLORMAP_RAINBOW)
    
    #no_matches = np.sum(mask_e_12)
    
    x = np.linspace(0,second_dist_max,51)
    sec_dist_kde_full = st.gaussian_kde(second_distances,bw_method='silverman')
    sec_dist_kde = sec_dist_kde_full(x)
        
    if plotMatches:
        full_map_BGR = cv2.applyColorMap(np.linspace(0,255,256).astype(np.uint8),cv2.COLORMAP_RAINBOW)
        full_map_RGB = cv2.cvtColor(full_map_BGR, cv2.COLOR_BGR2RGB)
        full_map = full_map_RGB[:,0,:] /255
        cmap= mpl.colors.ListedColormap(full_map)
        norm= mpl.colors.Normalize(vmin=0,vmax=second_dist_max)
        print("Second dist max ", second_dist_max)
        
        det_des_string = "Det: {} Des: {}".format(det_name, des_name)
        kp_img_0 = image_0
        
        if TILE_KP:
            feat_string_0 = "{}\nBefore tiling:{:d} after tiling {:d}".format(det_des_string, len(detector_kp_0), len(kp_0))
    #        kp_img_0 = draw_markers(kp_img_0, detector_kp_0, color=marker_colors, markerSize=1)
            
        else:            
            feat_string_0 = "{}\n{:d} features".format(det_des_string, len(kp_0))
    
        kp_img_0 = draw_markers(kp_img_0, kp_0, color=marker_colors)
    
        #fig1 = plt.figure(1); plt.clf()
        fig1, fig1_axes = plt.subplots(3,1, figsize= [9.28, 12])
        fig1.suptitle(settings['set_title'] + ' features')
        fig1_axes[0].axis("off"); fig1_axes[0].set_title(feat_string_0)
        fig1.canvas.draw()
        plt.pause(.1)
        fig1.set_size_inches([9.28, 12])
        fig1.subplots_adjust(top=0.91)
        #fig1.canvas.draw()
        #plt.pause(.1)
        fig1_axes[0].imshow(cv2.cvtColor(kp_img_0, cv2.COLOR_BGR2RGB)) 

        #cb = matplotlib.colorbar.ColorbarBase(fig1_axes[2], orientation='horizontal', 
        #                           cmap=cmap, norm=norm, fraction=1)
        #plt.pause(.1)
        cb_ax = adjust_lower_axes_to_image(fig1_axes)
                
        marker_colors_RGB = cv2.cvtColor(np.array(np.array(marker_colors)[:,np.newaxis,:],dtype=np.uint8), cv2.COLOR_BGR2RGB)[:,0,:]        

        if len(kp_eig_vals) != len(kp_0):
            color_eig_lookup = {}
            for k, c, e in zip(provided_kps, marker_colors_RGB, kp_eig_vals):
                color_eig_lookup[k.pt] = (c,e)
            kp_0_colors = []
            kp_0_eig_vals = []
            for k in kp_0:
                c, e = color_eig_lookup.get(k.pt, (np.array([255,255,255]), np.array([0,0])))
                kp_0_colors.append(c)
                kp_0_eig_vals.append(e)
            marker_colors_RGB = np.array(kp_0_colors)
            kp_eig_vals = np.array(kp_0_eig_vals)
        
        
        
        if plot_eig_movement:
            arrow_size = np.max(kp_eig_vals) * .01
            for (x1, y1), (x2,y2), c in zip(kp_eig_vals_old, kp_eig_vals, marker_colors_RGB):
                fig1_axes[1].scatter(kp_eig_vals[:,0], kp_eig_vals[:,1], color=(1, 1, 1, 0.0))
                arrow_col =  tuple(c/255) + tuple([.25])
                fig1_axes[1].arrow(x1, y1, x2-x1, y2-y1, fc=arrow_col, ec=arrow_col,
                                   head_width= arrow_size, head_length=2*arrow_size, overhang=arrow_size/4,
                                   length_includes_head=True, head_starts_at_zero=False)
            
        else:                
            fig1_axes[1].scatter(kp_eig_vals[:,0], kp_eig_vals[:,1], color=marker_colors_RGB/255, marker='+')

        if eig_data_lims is None:
            eig_data_lims = np.mean(kp_eig_vals,axis=0) + 1 * np.std(kp_eig_vals,axis=0)
        print("Eig value plot data lims: ",eig_data_lims)
        fig1_axes[1].set_xlim([0, eig_data_lims[1]])
        fig1_axes[1].set_ylim([0, eig_data_lims[1]])
        fig1_axes[1].set_aspect('equal', adjustable='datalim',anchor='SW')
        #fig1.canvas.draw()
        #fig1.canvas.flush_events()
        plt.pause(.1)
        
        fig1_axes[1].set_xlim([0, fig1_axes[1].get_xlim()[1]-fig1_axes[1].get_xlim()[0]])
        fig1_axes[1].set_ylim([0, fig1_axes[1].get_ylim()[1]-fig1_axes[1].get_ylim()[0]])

        fig1_axes[1].set_xlabel(r'$\lambda_1$'); fig1_axes[1].set_ylabel(r'$\lambda_2$')

        
        fig1_axes[2].hist(second_distances, bins=x, color='blue', density=True, alpha=0.4, label='Raw')
        fig1_axes[2].fill_between(x, sec_dist_kde, color='red',alpha=0.4)        
        fig1_axes[2].set_xlim((0,second_dist_max))
        fig1_axes[2].set_xticklabels([])
        fig1_axes[2].set_ylim([0,max_prob])
        fig1_axes[2].set_ylabel('Prob. density')
    
        #im=mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        #fig1.colorbar(im, ax=fig1_axes[1], cmap=cmap, orientation='horizontal', norm=norm)

        #cb_ax = fig1.add_axes([left0, 0.08+adjust_middle_axes_up + adjust_lower_axes_up, width0, 0.025])
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap,
                                       norm=norm, orientation='horizontal')
        cb_ax.set_xlabel('Distance to second closest matches')
        
        fig_ttl = fig1._suptitle.get_text() + '_' + fig1_axes[0].get_title()
        fig_ttl = fig_ttl.replace('$','').replace('\n','_').replace(' ','_')
        fig_fname = re.sub(r"\_\_+", "_", fig_ttl)

        if saveFig:
            save_fig2png(fig1, fname= fig_fname, size = [9.28, 12])
    
    result = {'detector':det_name, 'descriptor':des_name,
              'second_dist_kde': sec_dist_kde, 'kde_x':x, 'img0_no_features': len(kp_0), 
              'keypoints':kp_0}
    return result