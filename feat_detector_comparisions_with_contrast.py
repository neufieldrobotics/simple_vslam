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
from zernike.zernike import MultiHarrisZernike
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features
from feat_detector_comparisions_helper import *
import progressbar
#from datetime import datetime
import pandas as pd
sys.path.insert(0, os.path.abspath('./external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
from cmtpy import contrast_measurement as cm
from collections import deque

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


#K = np.array([[3523.90252470728501/5, 0.0, 2018.22833167806152/5],
#              [0.0, 3569.92180686745451/5, 1473.25249541175890/5],
#              [0.0, 0.0, 1.0]])

#D = np.array([-2.81360302828763176e-01, 1.38000456840603303e-01, 4.87629635176304053e-05, -6.01560125682630380e-05, -4.34666626743886730e-02])

raw_img_folder = os.path.join(path,raw_sets_folder)
clahe_img_folder = os.path.join(path,clahe_sets_folder)

raw_image_names = sorted(glob.glob(raw_img_folder+'/*.png'))[:7]
clahe_image_names = sorted(glob.glob(clahe_img_folder+'/*.tif'))[:7]

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
base_line_steps = [2,3]#[10,15] #[1, 2, 5, 10, 15, 20]
contrast_adj_factors = np.arange(0,-1.1,-.1)

image_names = raw_image_names

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
                            'baseline': base_line } 
        print("Proccessing pair: {} and {}".format(image_0_dict['name'], image_1_dict['name']))
        
        for i in range(len(contrast_adj_factors)):
            image_0 = image_0_dict['contrast_imgs'][i]
            image_1 = image_1_dict['contrast_imgs'][i]
            image_0_cms = {'img0_'+nm:cm for nm,cm in image_0_dict['contrast_measurements'][i].items() }
            image_1_cms = {'img1_'+nm:cm for nm,cm in image_1_dict['contrast_measurements'][i].items() }
 
            pair_config = {**pair_base_config, **image_0_cms, **image_1_cms,
                           'contrast_adj_factor': contrast_adj_factors[i] }
            
            for config_settings in config_settings_list:
                pair_results = analyze_image_pair(image_0, image_1, config_settings, plotMatches = False)  
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
    
output_file = "matching_results"+time.strftime("_%Y%m%d_%H%M%S")+".csv"
results_df.to_csv(output_file)
print("Results written to {}".format(output_file))


'''
  
for img_name in image_names[:-min(base_line_steps)]: 

    results_list = []

    image_names = raw_image_names
    
    #for img in image_names[:-BASELINE_STEP_SIZE]
    
    for img0_name, img1_name in progressbar.progressbar(zip(image_names[:-BASELINE_STEP_SIZE], image_names[BASELINE_STEP_SIZE:]),
                                                        max_value=len(image_names[:-BASELINE_STEP_SIZE])):
        image_0 = cv2.imread(img0_name, cv2.IMREAD_GRAYSCALE)
        image_1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)
        
        
       # ce(warped_images[i]) for nm, ce in contrast_estimators.items()]

        pair_config = {'set_title': base_settings['set_title'],
                       'image_0': os.path.splitext(os.path.basename(img0_name))[0], 
                       'image_1': os.path.splitext(os.path.basename(img1_name))[0], 
                       'contrast_adj_factor': 0,
                       'baseline': BASELINE_STEP_SIZE}
        
        pair_config.update({'img0_'+nm:ce(image_0) for nm, ce in contrast_estimators.items()})
        pair_config.update({'img1_'+nm:ce(image_1) for nm, ce in contrast_estimators.items()})
        
        pair_results = analyze_image_pair(image_0, image_1, config_settings, plotMatches = False)  
        pair_results.update(pair_config)
        results_df = results_df.append(pair_results, ignore_index=True)


    results_array = np.array(results_list)
    np.savetxt("results_array_baseline_"+str(BASELINE_STEP_SIZE)+'_'+datetime.now().strftime("%Y%m%d%H%M%S")+".csv", results_array, delimiter=",", header="zernike, surf, orb_sf")

    fig3 = plt.figure(3)
    plt.clf()
    bins = np.linspace(10, 250, 25)

    plt.hist(results_array, bins=bins, alpha=0.5, label=['Zernike','SURF','ORB_SF'])
    plt.legend(loc='upper right')
    plt.suptitle(config_settings['set_title'] + '\n Baseline: {:d}'.format(BASELINE_STEP_SIZE))
    plt.xlabel('Bins (Number of matches)')
    plt.ylabel('Occurances (Image pairs)')
    plt.axes().set_ylim([0, 750])
    plt.draw()
    #save_fig2pdf(fig3)
    
'''