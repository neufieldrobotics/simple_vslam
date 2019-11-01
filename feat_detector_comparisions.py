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
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,2) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1],
                ...,
                [xn,yn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keept or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter


def tiled_features(kp, img, tiley, tilex):
    feat_per_cell = int(len(kp)/(tilex*tiley))
    HEIGHT, WIDTH = img.shape
    assert WIDTH%tiley == 0, "Width is not a multiple of tilex"
    assert HEIGHT%tilex == 0, "Height is not a multiple of tiley"
    w_width = int(WIDTH/tiley)
    w_height = int(HEIGHT/tilex)
        
    xx = np.linspace(0,HEIGHT-w_height,tilex,dtype='int')
    print(xx)
    yy = np.linspace(0,WIDTH-w_width,tiley,dtype='int')
    print(yy)
        
    kps = np.array([])
    pts = np.array([keypoint.pt for keypoint in kp])
    kp = np.array(kp)
    
    for ix in xx:
        for iy in yy:
            inbox_mask = bounding_box(pts, iy,iy+w_height, ix,ix+w_height)
            inbox = kp[inbox_mask]
            inbox_sorted = sorted(inbox, key = lambda x:x.response, reverse = True)
            inbox_sorted_out = inbox_sorted[:feat_per_cell]
            print(np.shape(inbox), ' and ', np.shape(inbox_sorted_out))
            kps = np.append(kps,inbox_sorted_out)
            #print(kps)
    return kps.tolist()

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
orb_detector = cv2.ORB_create(nfeatures=1000, edgeThreshold=125, patchSize=125, nlevels=6, 
                              fastThreshold=9, scaleFactor=1.2, WTA_K=2,
                              scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike_detector = MultiHarrisZernike(Nfeats= 600, seci= 4, secj= 3, levels= 6, ratio= 1/1.2, 
                                      sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3)

sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures = 600, nOctaveLayers = 3, contrastThreshold = 0.01, 
                                            edgeThreshold = 20, sigma = 1.6)

raw_images = read_image_list(raw_image_names, resize_ratio=1/5)
clahe_images = read_image_list(clahe_image_names, resize_ratio=1/5)

zernike_kp_0, zernike_des_0 = zernike_detector.detectAndCompute(raw_images[0], mask=None, timing=False)
zernike_kp_1, zernike_des_1 = zernike_detector.detectAndCompute(raw_images[1], mask=None, timing=False)
orb_kp_0, orb_des_0 = orb_detector.detectAndCompute(raw_images[0], None)
orb_kp_1, orb_des_1 = orb_detector.detectAndCompute(raw_images[1], None)
sift_kp_0, sift_des_0 = sift_detector.detectAndCompute(raw_images[0], None)
sift_kp_1, sift_des_1 = sift_detector.detectAndCompute(raw_images[1], None)

zernike_kp_0_sort = sorted(zernike_kp_0, key = lambda x: x.response, reverse=True)
zernike_kp_1_sort = sorted(zernike_kp_1, key = lambda x: x.response, reverse=True)
orb_kp_0_sort = sorted(orb_kp_0, key = lambda x: x.response, reverse=True)
orb_kp_1_sort = sorted(orb_kp_1, key = lambda x: x.response, reverse=True)
sift_kp_0_sort = sorted(sift_kp_0, key = lambda x: x.response, reverse=True)
sift_kp_1_sort = sorted(sift_kp_1, key = lambda x: x.response, reverse=True)

zernike_kp_img_0 = cv2.drawKeypoints(raw_images[0], zernike_kp_0_sort[:25], cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB),color=[255,255,0],
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
zernike_kp_img_1 = cv2.drawKeypoints(raw_images[1], zernike_kp_1_sort[:25], cv2.cvtColor(raw_images[1], cv2.COLOR_GRAY2RGB),color=[255,255,0],
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
orb_kp_img_0 = cv2.drawKeypoints(raw_images[0], orb_kp_0_sort[:25], cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB),color=[255,255,0],
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
orb_kp_img_1 = cv2.drawKeypoints(raw_images[1], orb_kp_1_sort[:25], cv2.cvtColor(raw_images[1], cv2.COLOR_GRAY2RGB),color=[255,255,0],
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_kp_img_0 = cv2.drawKeypoints(raw_images[0], sift_kp_0_sort, cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB),color=[255,255,0],
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_kp_img_1 = cv2.drawKeypoints(raw_images[1], sift_kp_1_sort, cv2.cvtColor(raw_images[1], cv2.COLOR_GRAY2RGB),color=[255,255,0],
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
fig1_axes[0,2].imshow(sift_kp_img_0)
fig1_axes[1,2].axis("off"); fig1_axes[0,2].set_title("Sift Features")
fig1_axes[1,2].imshow(sift_kp_img_1)
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


sift_matches_01 = knn_match_and_lowe_ratio_filter(matcher_norm, sift_des_0, sift_des_1, threshold=0.90)

sift_kp0_match_01 = np.array([sift_kp_0[mat.queryIdx].pt for mat in sift_matches_01])
sift_kp1_match_01 = np.array([sift_kp_1[mat.trainIdx].pt for mat in sift_matches_01])

sift_kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(sift_kp0_match_01,axis=1),K,D)
sift_kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(sift_kp1_match_01,axis=1),K,D)

sift_E_12, sift_mask_e_12 = cv2.findEssentialMat(sift_kp0_match_01_ud, sift_kp1_match_01_ud, focal=1.0, pp=(0., 0.), 
                                               method=cv2.RANSAC, prob=0.9999, threshold=0.001)

print("sift After essential: ", np.sum(sift_mask_e_12))

sift_valid_matches_img = draw_matches_vertical(raw_images[0],sift_kp_0, raw_images[1],sift_kp_1, sift_matches_01, 
                                              sift_mask_e_12, display_invalid=True, color=(0, 255, 0))


fig2, fig2_axes = plt.subplots(1,3)
fig2.suptitle('800x600 Raw Images Feature Matching')
fig2_axes[0].axis("off"); fig2_axes[0].set_title("Zernike Features\n{:d} matches".format(np.sum(zernike_mask_e_12)))
fig2_axes[0].imshow(zernike_valid_matches_img)
fig2_axes[1].axis("off"); fig2_axes[1].set_title("Orb Features\n{:d} matches".format(np.sum(orb_mask_e_12)))
fig2_axes[1].imshow(orb_valid_matches_img)
fig2_axes[2].axis("off"); fig2_axes[2].set_title("Sift Features\n{:d} matches".format(np.sum(sift_mask_e_12)))
fig2_axes[2].imshow(sift_valid_matches_img)
fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)
