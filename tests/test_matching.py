#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:11:54 2019

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
from helper_functions.frame import Frame
from vslam_helper import *


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = path+'data/time_lapse_5_cervino_800x600/G0057821.png'
img2 = path+'data/time_lapse_5_cervino_800x600/G0057826.png'

Frame.detector = MultiHarrisZernike(Nfeats=1200,like_matlab=False)
Frame.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
Frame.config_dict= {'lowe_ratio_test_threshold': 0.9}
Frame.K = np.array([[699.33112889, 0.0, 403.23876197],
                    [0.0, 693.66457792, 293.40739086],
                    [0.0, 0.0, 1.0]])

Frame.D = np.array([-2.78089511e-01,  1.30037134e-01, -1.17555797e-04, -1.55337290e-04, -4.34486330e-02])

Frame.is_config_set = True

fr1=Frame(img1)
fr2=Frame(img2)

Frame.match_and_propagate_keypoints(fr1, fr2, initialization=True)

#kp1_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr1.kp[fr1.kp_cand_ind],1), Frame.K, Frame.D)[:,0,:]
#kp2_m_prev_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr2.kp[fr2.kp_m_prev_cand_ind],1), Frame.K, Frame.D)[:,0,:]

kp1_cand_pt_ud = fr1.kp_ud[fr1.kp_cand_ind]

kp2_m_prev_cand_pt_ud = fr2.kp_ud[fr2.kp_m_prev_cand_ind]

E_12, mask_e_12 = cv2.findEssentialMat(kp1_cand_pt_ud, 
                                       kp2_m_prev_cand_pt_ud,
                                       cameraMatrix=Frame.K, method=cv2.RANSAC, 
                                       prob= 0.9999, threshold= 0.8)

F_12, mask_F_12 = cv2.findFundamentalMat(kp1_cand_pt_ud, 
                                         kp2_m_prev_cand_pt_ud,
                                         method=cv2.FM_RANSAC, 
                                         confidence= 0.9999, ransacReprojThreshold= 0.8)

print("Fund matrix: used {} of total {} matches".format(np.sum(mask_F_12),len(kp2_m_prev_cand_pt_ud)))
essen_mat_pts = np.sum(mask_F_12)

            
img12 = draw_point_tracks(fr1.kp[fr1.kp_cand_ind], fr2.gr,
                          fr2.kp[fr2.kp_m_prev_cand_ind],
                          mask_F_12[:,0].astype(bool), False, color=[255,255,0])

fig, ax= plt.subplots(dpi=200)
plt.title('Multiscale Harris with Zernike Angles')
plt.axis("off")
plt.imshow(img12)
plt.show()

lines = cv2.computeCorrespondEpilines(fr1.kp[fr1.kp_cand_ind[mask_F_12[:,0].astype(bool)]], whichImage=1, 	F=F_12) 

def line_end_pts(lines, x=np.array([0,799])):
    '''
    Lines corresponding to the points in the other image. Each line ax+by+c=0 is encoded by 3 numbers (a,b,c) .

    '''
    y = (-lines[:,[2]] - lines[:,[0]]*x)/ lines[:,[1]]
    pts_start = np.ones_like(y) * x[0]
    pts_start[:,1] = y[:,0]
    
    pts_end = np.ones_like(y) * x[1]
    pts_end[:,1] = y[:,1]

    return pts_start, pts_end

pt1, pt2 = line_end_pts(lines[:,0,:])

for p1,p2 in zip(pt1,pt2):
    cv2.line(img12, (int(p1[0]),int(p1[1])),
                    (int(p2[0]),int(p2[1])), (0, 255, 0))

fig, ax= plt.subplots(dpi=200)
plt.title('Multiscale Harris with Zernike Angles')
plt.axis("off")
plt.imshow(img12)
plt.show()

pts = np.array([[2,0],[2,2],[0,2]])
lines = np.array([[1,1,-1],[1,-2,-1]])

ab = lines[:,:2]
c = lines[:,2]
num = np.abs(pts @ ab + c)
den = np.linalg.norm(ab, axis=1)
dist = num / den
