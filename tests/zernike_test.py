#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:52:23 2019

@author: vik748
"""
from zernike import MultiHarrisZernike
import sys
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
#from vslam_helper import *


def knn_match_and_filter(matcher, kp1, kp2, des1, des2,threshold=0.9):
    matches_knn = matcher.knnMatch(des1,des2, k=2)
    matches = []
    kp1_match = []
    kp2_match = []
    
    for i,match in enumerate(matches_knn):
        if len(match)>1:
            if match[0].distance < threshold*match[1].distance:
                matches.append(match[0])
                kp1_match.append(kp1[match[0].queryIdx].pt)
                kp2_match.append(kp2[match[0].trainIdx].pt)
        elif len(match)==1:
            matches.append(match[0])
            kp1_match.append(kp1[match[0].queryIdx].pt)
            kp2_match.append(kp2[match[0].trainIdx].pt)

    return np.ascontiguousarray(kp1_match), np.ascontiguousarray(kp2_match), matches


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057821.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
img2 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057826.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img2 = cv2.imread(path+'data/skerki_small/all/ESC.970622_025513.0622.tif',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

gr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

a = MultiHarrisZernike(Nfeats=600,like_matlab=False)
m1 = cv2.imread(path+'data/time_lapse_5_cervino_800x600_masks_out/G0057821_mask.png',cv2.IMREAD_GRAYSCALE)
m2 = cv2.imread(path+'data/time_lapse_5_cervino_800x600_masks_out/G0057826_mask.png',cv2.IMREAD_GRAYSCALE)

kp1, des1 = a.detectAndCompute(gr1, mask=m1, timing=True)
kp2, des2 = a.detectAndCompute(gr2, mask=m2, timing=True)

outImage	 = cv2.drawKeypoints(gr1, kp1, gr1,color=[255,255,0],
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
fig, ax= plt.subplots(dpi=200)
plt.title('Multiscale Harris with Zernike Angles')
plt.axis("off")
plt.imshow(outImage)
plt.show()

kp1_pts = np.array([k.pt for k in kp1],dtype=np.float32)

#kp2_pts, mask_klt, err = cv2.calcOpticalFlowPyrLK(gr1, gr2, kp1_pts, None, **config_dict['KLT_settings'])
#print ("KLT tracked: ",np.sum(mask_klt) ," of total ",len(kp1_pts),"keypoints")

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


a,b,matches = knn_match_and_filter(matcher, kp1, kp2, des1, des2,threshold=0.9)



def track_keypoints(kp1, des1, kp2, des2, matches):
    '''
    track_keypoints accepts 2 list of keypoints and descriptors and a list of 
    matches and returns, matched lists of keypoints and descriptors. Also returns
    a set of candidate keypoints from 2 which didn't have matches in 1
    
    Parameters
    ----------
    kp1 : list of cv2.KeyPoint objects
        Keypoints from frame 1
    des1 : NxD array
        NxD array of descriptors where N = len(kp1) and D is length of descriptor
    kp2 : list of cv2.KeyPoint objects
        Keypoints from frame 2
    des2 : NxD array
        NxD array of descriptors where N = len(kp2) and D is length of descriptor        
    matches : list of cv2.DMatch objects
        List of matches between 1 and 2. (single matches only not KNN)
    '''
    kp1_matched = []    
    kp2_matched = []
    des1_matched = np.zeros((len(kp1),des1.shape[1]),des1.dtype)
    des2_matched = np.zeros((len(kp2),des2.shape[1]),des2.dtype)
    matched_kp2_indx = []

    # Go through matches and create subsets of kp1 and kp2 which matched
    for i,m in enumerate(matches):
        kp1_matched += [kp1[m.queryIdx]]
        kp2_matched += [kp2[m.trainIdx]]
        des1_matched[i,:] = des1[m.queryIdx,:]
        des2_matched[i,:] = des2[m.trainIdx,:]
        matched_kp2_indx += [m.trainIdx]

    kp2_cand=[]
    cand_indx = []
    for i,k in enumerate(kp2):
        if i not in matched_kp2_indx:
            kp2_cand += [k]
            cand_indx += [i]
    des2_cand = des2[cand_indx,:]
    
    return kp1_matched, des1_matched, kp2_matched, des2_matched, kp2_cand, des2_cand

def track_matched_kp_and_filter_cand(kp1,kp2, matches):
    m_qid_vals = []
    kp1_pts = np.zeros((len(matches),1,2),dtype='float32')
    kp2_pts = np.zeros((len(matches),1,2),dtype='float32')
    # Create dictionary of kp1 to kp2 indexes and list of matched kp2 indexes
    for i,m in enumerate(matches):
        kp1_pts[i,1,:] = kp1[m.queryIdx].pt
        kp2_pts[i,1,:] = kp2[m.trainIdx].pt
        m_qid_vals += [m.trainIdx]
        
    kp2_cand=[]    
    for i,v in enumerate(kp2):
        if i not in m_qid_vals:
            kp2_cand += [v]
            
    
    #kp_pts = np.expand_dims(np.array([o.pt for o in kp],dtype='float32'),1)
    
    return kp1_pts, kp2_pts, kp2_cand_pts