#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:52:23 2019

@author: vik748
"""
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
#from vslam_helper import *
from zernike.zernike import MultiHarrisZernike
from matplotlib import pyplot as plt


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
a.plot_zernike(a.ZstrucZ)


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


_,_,matches = knn_match_and_filter(matcher, kp1, kp2, des1, des2,threshold=0.9)