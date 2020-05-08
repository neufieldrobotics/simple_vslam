#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
from zernike.zernike import MultiHarrisZernike
from helper_functions.frame import Frame

import sys
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)8s  %(message)s')
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
    
img_name = path+'data/time_lapse_5_cervino_800x600/G0057821.png'

a = MultiHarrisZernike(Nfeats=600,like_matlab=False)

Frame.K = np.eye(3)
Frame.D = np.zeros([5,1])
Frame.detector = a
Frame.is_config_set = True

fr1 = Frame(img_name)
fr1.show_features()