#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
import numpy as np
np.set_printoptions(precision=5,suppress=True)
import cv2

class Frame ():
    '''
 
    '''
    def __init__(self, gr, mask, kp, des, K, D):       
        self.gr      = gr        # the grayscale image
        self.mask    = mask      # mask for the image
        self.kp      = kp        # Image keypoint locations
        self.des     = des       # Keypoint descriptors
        self.K       = K
        self.D       = D
        self.kp_ud   = cv2.undistortPoints(np.expand_dims(self.kp,1),
                                           self.K,self.D)[:,0,:]



