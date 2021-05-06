#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,'external_packages','zernike_py'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,'helper_functions'))
from frame import Frame

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

data_path = os.path.join(os.pardir,'data')

img_name = os.path.join(data_path, 'cervino_800x600/G0057821.jpg')

a = MultiHarrisZernike(Nfeats=100,like_matlab=False)

Frame.K = np.eye(3)
Frame.D = np.zeros([5,1])
Frame.detector = a
Frame.descriptor = a
Frame.is_config_set = True

fr1 = Frame(img_name)
fr1.show_features()
