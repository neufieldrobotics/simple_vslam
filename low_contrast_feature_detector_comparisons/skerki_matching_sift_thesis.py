#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo to show MultiHarrisZernike feature detector on the Skerki dataset

@author: vik748
"""
import sys, os
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath('../external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike

def knn_match_and_lowe_ratio_filter(matcher, des1, des2,threshold=0.9, dist_mask_12=None, draw_plot_dist=False):
    # First match 2 against 1
    if dist_mask_12 is None:
        dist_mask_21 = None
    else:
        dist_mask_21 = dist_mask_12.T
    matches_knn = matcher.knnMatch(des2,des1, k=2, mask = dist_mask_21 )
    all_ds = [m[0].distance for m in matches_knn if len(m) >0]

    #print("Len of knn matches", len(matches_knn))

    matches = []
    # Run lowes filter and filter with difference higher than threshold this might
    # still leave multiple matches into 1 (train descriptors)
    # Create mask of size des1 x des2 for permissible matches
    mask = np.zeros((des1.shape[0],des2.shape[0]),dtype='uint8')
    for match in matches_knn:
        if len(match)==1 or (len(match)>1 and match[0].distance < threshold*match[1].distance):
           # if match[0].distance < 75:
                matches.append(match[0])
                mask[match[0].trainIdx,match[0].queryIdx] = 1

    #matches = [m for m in matches if m.distance<5 ]

    if draw_plot_dist:
        fig, axes = plt.subplots(1, 1, num=3)
        filt_ds = [m.distance for m in matches]
        axes.plot(sorted(all_ds),'.',label = 'All Matches')
        axes.plot(sorted(filt_ds),'.',label = 'Filtered Matches')
        axes.set_xlabel('Number')
        axes.set_ylabel('Distance')
        axes.legend()
        plt.pause(.1)

    # run matches again using mask but from 1 to 2 which should remove duplicates
    # This is basically same as running cross match after lowe ratio test
    matches_cross = matcher.match(des1,des2,mask=mask)
    #print("Len of cross matches", len(matches_cross))
    return matches_cross

def draw_arrows(vis_orig, points1, points2, color = (0, 255, 0), thick = 2, tip_length = 0.25):
    if len(vis_orig.shape) == 2: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    else: vis = vis_orig
    for p1,p2 in zip(points1,points2):
        cv2.arrowedLine(vis, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),
                        color=color, thickness=thick, tipLength = tip_length)
    return vis

def draw_feature_tracks(img_left,kp1,img_right,kp2, matches, mask, display_invalid=False, color=(0, 255, 0), thick = 2):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    bool_mask = mask.astype(bool)
    valid_right_matches = np.array([kp2[mat.trainIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    valid_left_matches = np.array([kp1[mat.queryIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    #img_right_out = draw_points(img_right, valid_right_matches)
    img_right_out = draw_arrows(img_right, valid_left_matches, valid_right_matches, thick = thick)
    img_left_out = draw_arrows(img_left, valid_right_matches, valid_left_matches, thick = thick)

    return img_left_out, img_right_out


#img0_name = os.path.join('test_data', 'skerki_test_image_0.tif')
#img1_name = os.path.join('test_data', 'skerki_test_image_1.tif')

img0_name = '/home/vik748/data/low_contrast_datasets/skerki_mud/skerki_mud_RAW/ESC.970622_024911.0595.tif'
img1_name = '/home/vik748/data/low_contrast_datasets/skerki_mud/skerki_mud_RAW/ESC.970622_030745.0679.tif'
#img1_name = '/home/vik748/data/low_contrast_datasets/skerki_mud/skerki_mud_RAW/ESC.970622_030758.0680.tif'

#img0_name = '/home/vik748/data/low_contrast_datasets/skerki_mud_CLAHE/skerki_mud_CLAHE_RAW/ESC.970622_024911.0595.tif'
#img1_name = '/home/vik748/data/low_contrast_datasets/skerki_mud_CLAHE/skerki_mud_CLAHE_RAW/ESC.970622_030745.0679.tif'
#img1_name = '/home/vik748/data/low_contrast_datasets/skerki_mud_CLAHE/skerki_mud_CLAHE_RAW/ESC.970622_030758.0680.tif'


gr0 = cv2.imread(img0_name, cv2.IMREAD_GRAYSCALE)
gr1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)

detectpr = MultiHarrisZernike(Nfeats= 1200, seci = 5, secj = 4, levels = 6, ratio = 0.75,
                                 sigi = 2.75, sigd = 1.0, nmax = 8, like_matlab=False, lmax_nd = 3, harris_threshold = None)

detector = cv2.xfeatures2d.SIFT_create(nfeatures = 2400, nOctaveLayers = 6, contrastThreshold = 0.001,
                                       edgeThreshold = 20, sigma = 1.6)


#zernike_obj.plot_zernike(zernike_obj.ZstrucZ)

kp0, des0 = detector.detectAndCompute(gr0, mask=None)
kp1, des1 = detector.detectAndCompute(gr1, mask=None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

matches_01 = knn_match_and_lowe_ratio_filter(matcher, des0, des1, threshold=0.85)

kp0_match_01_pts = np.array([kp0[mat.queryIdx].pt for mat in matches_01])
kp1_match_01_pts = np.array([kp1[mat.trainIdx].pt for mat in matches_01])

'''

if [int(x) for x in cv2.__version__.split(".")] >= [3,4,0]:
    E_12, mask_e_12 = cv2.findFundamentalMat(kp0_match_01_pts, kp1_match_01_pts,
                                             method=cv2.FM_RANSAC,       # RAnsac
                                             ransacReprojThreshold=.75,  # Inlier threshold in pixel since we don't use nomalized coordinates
                                             confidence=0.999999) #maxIters = 10000000)
else:
    E_12, mask_e_12 = cv2.findFundamentalMat(kp0_match_01_pts, kp1_match_01_pts,
                                             method=cv2.FM_RANSAC, # Ransac
                                             param1=.75,           # Inlier threshold in pixel since we don't use nomalized coordinates
                                             param2=0.999999)
'''
retval, mask_e_12 = cv2.findHomography(kp0_match_01_pts, kp1_match_01_pts,
                                       method = cv2.RANSAC,
                                       ransacReprojThreshold = 3.0,
                                       maxIters = 100000, confidence = 0.99999)

kp0_match_inliers = [kp0[mat.queryIdx] for mat, msk in zip(matches_01, mask_e_12) if msk]
kp1_match_inliers = [kp1[mat.trainIdx] for mat, msk in zip(matches_01, mask_e_12) if msk]

gr0_inliers = cv2.drawKeypoints(gr0, kp0_match_inliers, gr0,color=[255,255,0],
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
gr1_inliers = cv2.drawKeypoints(gr1, kp1_match_inliers, gr0,color=[255,255,0],
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

matches_img_left, matches_img_right = draw_feature_tracks(gr0_inliers, kp0, gr1_inliers, kp1, matches_01,
                                                          mask_e_12, display_invalid=True, color=(0, 255, 0),
                                                          thick = 2)

fig2, fig2_axes = plt.subplots(1,2)
fig2.suptitle('SIFT Feature Matching: {:d} matches'.format(np.sum(mask_e_12)))
[ax.axis("off") for ax in fig2_axes]
fig2_axes[0].imshow(matches_img_left)
fig2_axes[1].imshow(matches_img_right)
fig2.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=.9, wspace=0.1, hspace=0.0)

good_matches = [m for m,inlier in zip(matches_01, mask_e_12) if inlier==1]

mat_img = cv2.drawMatches(gr0,kp0,gr1,kp1,good_matches,None,matchColor=[255,255,0], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
fig3, fig3_axes = plt.subplots(1,1)
fig3_axes.axis("off")
fig3_axes.imshow(mat_img)
fig3_axes.set_title('SIFT Feature Matching: {:d} matches'.format(np.sum(mask_e_12)))
fig3.tight_layout()
fig3.set_size_inches([18, 6])
fig3.savefig("SIFT_matching_CLAHE_17_matches.png", dpi=300)