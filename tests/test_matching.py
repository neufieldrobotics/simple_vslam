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
#from helper_functions.frame import Frame
from vslam_helper import *

fr_i = fr_prev
fr_j = fr_curr

T = fr_curr.T_pnp # Target camera pose in world
lm_points_3d = Frame.landmarks[fr_i.lm_ind_non_match] # points in the world
#fr_j_kp = fr_curr.kp
max_matches = 5
search_radius = 25.0
#fr_j_kp_m_prev_lm = fr_curr.kp[fr_curr.kp_m_prev_lm_ind]

rvec, tvec = decompose_T(T_inv(T))
objectPoints = np.ascontiguousarray(lm_points_3d)

imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, Frame.K, Frame.D, aspectRatio = 0)

#print(imagePoints[:,0,:] - fr_curr.kp[fr_curr.kp_m_prev_lm_ind])
img12_cand = draw_points(Frame.img_cand_pts, imagePoints[:,0,:], color=[255,0,0])
plt.imshow(img12_cand)

FLANN_INDEX_KDTREE = 0
kd_tree = cv2.flann_Index(fr_j.kp, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 1})

mask = np.zeros((fr_i.des.shape[0],fr_j.des.shape[0]),dtype='uint8')


for pt in imagePoints:
    print (pt)
    #kp = fr_j.kp[indx]
    dist_inlier_indices = -np.ones((1,max_matches),dtype=np.int32)
    dist_inlier_dists = -np.ones((1,max_matches),dtype=np.float32)

    ret_val, dist_inlier_ind, distinliner_dists = kd_tree.radiusSearch(pt, search_radius, max_matches, 
                                                                       indices = dist_inlier_indices, dists=dist_inlier_dists)
    print(dist_inlier_ind)
#    for inlier in dist_inlier_ind[dist_inlier_ind > 0]:
#        mask[indx, inlier] = 1

'''
rvec, tvec = decompose_T(T_inv(fr2.T_pnp))
rj,_ = cv2.Rodrigues(rvec)
objectPoints = np.ascontiguousarray(Frame.landmarks)

imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, Frame.K, Frame.D, aspectRatio = 0)

print(imagePoints[:,0,:] - fr2.kp[fr2.kp_lm_ind])
#img12_cand = draw_points(Frame.img_cand_pts, imagePoints[:,0,:], color=[255,0,0])
#plt.imshow(img12_cand)

'''