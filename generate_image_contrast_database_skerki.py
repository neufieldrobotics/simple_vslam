#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import cv2
import sys
import os
import glob
from feat_detector_comparisions_helper import process_image_contrasts, read_grimage
import progressbar
#from datetime import datetime
import pandas as pd
import ipyparallel as ipp

if sys.platform == 'darwin':
    path = '/data/skerki'
else:
    path = os.path.expanduser('/data/skerki')
import time

#raw_sets_folder = 'Lars2_081018_800x600'
raw_sets_folder = 'full_4bit_cropped'
#raw_sets_folder = 'Stingray2_080718_800x600'

raw_img_folder = os.path.join(path,raw_sets_folder)
#mask_folder = raw_img_folder.replace('_800x600', '_masks_from_model_800x600')
mask_folder = None
ctrst_img_output_folder = raw_img_folder + "_ctrst_enhanced"
if not os.path.exists(ctrst_img_output_folder):
    os.makedirs(ctrst_img_output_folder)

raw_image_names = sorted(glob.glob(raw_img_folder+'/*.tif'))

base_settings = {'set_title': os.path.basename(raw_img_folder)}

#contrast_adj_factors = np.arange(0,-1.1,-.1)  #np.array([0.0, -0.5, -1.0])
contrast_adj_factors = np.array([1.0])

image_names = raw_image_names

tic = time.time()

ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, lam=5, adjustment_factor=-1.0, stretch_factor=-1.0,
                              min_stretch_bits=4, downsample_for_kde=True,debug=False, plot_histograms=False)

for img_name in image_names:
    img_name_base, img_name_ext = os.path.splitext(os.path.basename(img_name))
    img = read_grimage(img_name)    
        
    v_k, a_k = ace_obj.compute_vk_and_ak(img)
    
    warped_images = np.empty(contrast_adj_factors.shape,dtype=object)
    for i,adj in enumerate(contrast_adj_factors):
        if adj==0:
            warped_images[i] = img
        else:
            outputs = ace_obj.compute_bk_and_dk(v_k, a_k, adjustment_factor=adj, stretch_factor=adj)
            warped_images[i], Tx = ace_obj.transform_image(*outputs, img)
        out_img_name = os.path.join(ctrst_img_output_folder, img_name_base+"_ctrst_adj_{:.2f}.{}".format(adj,img_name_ext) )
        cv2.imwrite(out_img_name, warped_images[i])
    print ("{} written".format(out_img_name))

