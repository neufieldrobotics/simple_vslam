#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 HISTORY      WHO     WHAT
----------    ----    -------------------------------------
2021-02-14    vs      Started writing this file using 'feat_detector_comparisions_with_contrast_test' as a template
                      The goal of this file is to write a single pair feature descriptor distance analysis function
"""
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import glob
sys.path.insert(0, os.path.abspath('../external_packages/zernike_py/'))
from zernike_py.MultiHarrisZernike import MultiHarrisZernike
sys.path.insert(0, os.path.abspath('../external_packages/cmtpy/'))
from cmtpy.histogram_warping_ace import HistogramWarpingACE
sys.path.insert(0, os.path.abspath('../'))
from vslam_helper import knn_match_and_lowe_ratio_filter, draw_feature_tracks, tiled_features, draw_arrows
from feat_detector_comparisions_helper import *
import progressbar
from datetime import datetime
import scipy.stats as st
from feat_descriptor_comparison_setup import *

dataset_name = datasets[2]
contrast_type = contrast_types[0]


fig0, fig0_axes = plt.subplots(1,1)
fig0.suptitle(dataset_name)

for contrast_type in contrast_types:
    img0_name = image_names[dataset_name]
    if contrast_type != "RAW":
        img0_name = os.path.splitext(img0_name)[0] + '.png'
    img0_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
    
    
    config_settings = {'set_title': dataset_name+"_"+contrast_type, #'K':K, 'D':D, 
                       'TILE_KP':TILE_KP, 'tiling':tiling ,
                       'detector': orb, 'descriptor': orb, 'findFundamentalMat_params':findFundamentalMat_params,
                       'NO_OF_FEATURES': NO_OF_FEATURES }
    
    plot_display_settings = plot_display_settings_database.get((image_names[dataset_name],config_settings['descriptor']))
    if plot_display_settings is not None: 
        config_settings.update(plot_display_settings)
    
    image_0 = read_grimage(img0_filename)    
    
    results = analyze_descriptor_distance_image_pair(image_0, config_settings, plotMatches=False, saveFig=True)
    if results['kde_x'] is not None:
        fig0_axes.plot(results['kde_x'], results['second_dist_kde'], label=contrast_type)
    fig0_axes.set_title("Det: {} Des: {}".format(results['detector'], results['descriptor']))
    
fig0_axes.legend()
fig0_axes.set_xlim([0, None])
fig0_axes.set_xlabel("Distance to 2nd closest matches")
fig0_axes.set_ylim([0, None])
fig0_axes.set_ylabel("Kernel Density Estimate")
fig_ttl = fig0._suptitle.get_text() + '_' + fig0_axes.get_title()
fig_ttl = fig_ttl.replace('$','').replace('\n','_').replace(' ','_')
fig_fname = re.sub(r"\_\_+", "_", fig_ttl) 
save_fig2png(fig0, fname= fig_fname, size = [8, 4.875])
