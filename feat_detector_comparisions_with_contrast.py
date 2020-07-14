#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import os
import glob
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features
from feat_detector_comparisions_helper import *
import progressbar
#from datetime import datetime
import pandas as pd
sys.path.insert(0, os.path.abspath('./external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
from cmtpy import contrast_measurement as cm
sys.path.insert(0, os.path.abspath('./external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
from collections import deque
from itertools import tee

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('~/data/')
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

raw_sets_folder = 'Lars1_080818_800x600'
clahe_sets_folder = 'Lars1_080818_clahe_800x600'

TILE_KP = True
tiling = (4,3)
NO_OF_FEATURES = 600
BASELINE_STEP_SIZE = 10

if TILE_KP:
    NO_OF_UT_FEATURES = NO_OF_FEATURES * 2
else:
    NO_OF_UT_FEATURES = NO_OF_FEATURES

raw_img_folder = os.path.join(path,raw_sets_folder)
clahe_img_folder = os.path.join(path,clahe_sets_folder)

raw_image_names = sorted(glob.glob(raw_img_folder+'/*.png'))
clahe_image_names = sorted(glob.glob(clahe_img_folder+'/*.tif'))

'''
Detect Features
'''
orb = cv2.ORB_create(nfeatures = NO_OF_UT_FEATURES, edgeThreshold=31, patchSize=31, nlevels=6,
                     fastThreshold=1, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

zernike = MultiHarrisZernike(Nfeats= NO_OF_FEATURES, seci= 2, secj= 3, levels= 6, ratio= 1/1.2,
                             sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3)

surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 50, nOctaves = 6)

sift = cv2.xfeatures2d.SIFT_create(nfeatures = NO_OF_UT_FEATURES, nOctaveLayers = 3, contrastThreshold = 0.01,
                                   edgeThreshold = 20, sigma = 1.6)


findFundamentalMat_params = {'method':cv2.FM_RANSAC,       # RAnsac
                             'param1':1.0,                # Inlier threshold in pixel since we don't use nomalized coordinates
                             'param2':0.9999}              # Confidence level


base_settings = {'set_title': os.path.basename(raw_img_folder), 'findFundamentalMat_params':findFundamentalMat_params, 
                   'TILE_KP':TILE_KP, 'tiling':tiling }

config_settings_list = [{**base_settings, 'detector': zernike, 'descriptor': orb},
                        {**base_settings, 'detector': orb, 'descriptor': orb},
                        {**base_settings, 'TILE_KP':False, 'detector': zernike, 'descriptor': zernike} ]

results_df = pd.DataFrame(columns = ['set_title','image_0', 'image_1', 'contrast_adj_factor', 'baseline',
                                     'detector', 'descriptor', 'img0_no_features','img1_no_features', 'matches',
                                     'img0_global_contrast_factor', 'img0_rms_contrast', 'img0_local_box_filt','img0_local_gaussian_filt', 'img0_local_bilateral_filt',
                                     'img1_global_contrast_factor', 'img1_rms_contrast', 'img1_local_box_filt','img1_local_gaussian_filt', 'img1_local_bilateral_filt'])

# image_skip determines how pairs are compared 
# image_skip = 5 with baseline 10,15 would mean :> (0,10), (0,15),(5,15), (5,20), (10,20), (10,25)  etc

image_skip = 5
#[10,15,20] #[1, 2, 5, 10, 15, 20]
base_line_steps = np.divide([10,15,20], image_skip).astype(int).tolist()

contrast_adj_factors = np.array([0.0, -0.5, -1.0]) #np.arange(0,-1.1,-.1)

image_names = raw_image_names[0:25:image_skip]

if not os.path.exists('results'):
    os.makedirs('results')

# Fill queue
img_queue = deque(maxlen = max(base_line_steps) + 1 )

img_iter = iter(image_names)
while len(img_queue) <= max(base_line_steps):
    img_name = next(img_iter)
    img = read_grimage(img_name)
    contrast_imgs, contrast_meas = generate_contrast_images(img, contrast_adj_factors=contrast_adj_factors)
    img_dict = {'name': os.path.splitext(os.path.basename(img_name))[0], 
                'contrast_imgs': contrast_imgs,
                'contrast_measurements': contrast_meas}
    img_queue.append(img_dict)
    print("Added image {} to queue".format(img_name))

# These help to rerun the code without refilling the buffer
img_queue_orig = img_queue.copy()
img_iter, img_iter_copy = tee(img_iter)

img_queue = img_queue_orig.copy()
img_iter, img_iter_copy = tee(img_iter_copy)

progress_bar = progressbar.ProgressBar(max_value=len(image_names) - max(base_line_steps) )
img_count = 0
progress_bar.update(img_count)
while True:
#for next_image_name in img_iter:
    image_0_dict = img_queue[0]
    assert len(contrast_adj_factors) == len(image_0_dict['contrast_imgs']) == len(image_0_dict['contrast_measurements'])
    
    for base_line in base_line_steps:
        image_1_dict = img_queue[base_line]
        
        pair_base_config = {'set_title': base_settings['set_title'],
                            'image_0': image_0_dict['name'], 'image_1': image_1_dict['name'],
                            'baseline': base_line * image_skip} 
        print("Proccessing pair: {} and {}".format(image_0_dict['name'], image_1_dict['name']))
        
        for i in range(len(contrast_adj_factors)):
            image_0 = image_0_dict['contrast_imgs'][i]
            image_1 = image_1_dict['contrast_imgs'][i]
            image_0_cms = {'img0_'+nm:cm for nm,cm in image_0_dict['contrast_measurements'][i].items() }
            image_1_cms = {'img1_'+nm:cm for nm,cm in image_1_dict['contrast_measurements'][i].items() }
 
            pair_config = {**pair_base_config, **image_0_cms, **image_1_cms,
                           'contrast_adj_factor': contrast_adj_factors[i] }
                       
            for config_settings in config_settings_list:
                config_settings_2 = {**config_settings, 'set_title':config_settings['set_title']+" Ctrst fact: {:.1f}".format(contrast_adj_factors[i])}

                pair_results = analyze_image_pair(image_0, image_1, config_settings_2, 
                                                  plotMatches = False, saveFig = False)  
                pair_results.update(pair_config)
                results_df = results_df.append(pair_results, ignore_index=True)
    
    img_queue.popleft()
    img_count += 1
    progress_bar.update(img_count); #sys.stdout.flush()

    try:
        next_image_name = next(img_iter)
    except StopIteration:
        break

    img = read_grimage(next_image_name)
    contrast_imgs, contrast_meas = generate_contrast_images(img, contrast_adj_factors=contrast_adj_factors)
    img_dict = {'name': os.path.splitext(os.path.basename(next_image_name))[0], 
                'contrast_imgs': contrast_imgs,
                'contrast_measurements': contrast_meas}
    img_queue.append(img_dict)
    

output_file = "results/matching_results"+time.strftime("_%Y%m%d_%H%M%S")+".csv"
results_df.to_csv(output_file)
print("Results written to {}".format(output_file))