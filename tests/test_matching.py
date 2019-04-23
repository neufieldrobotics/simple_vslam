#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:11:54 2019

@author: vik748
"""
des1 = fr1.des
des2 = fr2.des
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# First match 2 against 1
matches_knn = matcher.knnMatch(des2,des1, k=2)

matches = []

# Run lowes filter and filter with difference higher than threshold this might
# sill leave multiple matches into 1 (train descriptors)
# Create mask of size des1 x des2 for permissible matches
mask = np.zeros((des1.shape[0],des2.shape[0]),dtype='uint8')
for match in matches_knn:
    if len(match)==1 or (len(match)>1 and match[0].distance < 0.9*match[1].distance):
            matches.append(match[0])
            mask[match[0].trainIdx,match[0].queryIdx] = 1

# run matches again using mask but from 1 to 2 which should remove duplicates            
matches_cross = matcher.match(des1,des2,mask=mask)

des1_ind = []
des2_ind = []


'''
# Go through matches and create list of indices of kp1 and kp2 which matched
for i,m in enumerate(matches):
    des1_ind += [m.queryIdx]
    des2_ind += [m.trainIdx]
    
des1 = des1[des1_ind]
des2 = des2[des2_ind]
'''