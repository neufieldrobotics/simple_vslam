#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 HISTORY      WHO     WHAT
----------    ----    -------------------------------------
2021-02-25    vs      Started writing this file using 'feat_detector_comparisions_with_contrast_test' as a template
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
from scipy.fft import fft, fftfreq, fft2
from matplotlib.colors import LogNorm


#datasets_ind = [2,3,6,4,5]
datasets_ind = [12,13,16,14,15]


plt.close('all')
fig, axes = plt.subplots(2,len(datasets_ind),figsize=[18.41,  6.55])
[ax.axis('off') for ax in axes.ravel()]

for i, ind in enumerate(datasets_ind):
    dataset_name = datasets[ind]
    contrast_type = contrast_types[0]

    img0_name = image_names[dataset_name]
    img0_name_base, image0_name_ext = os.path.splitext(img0_name)
    img0_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
    img0_mask_filename = os.path.join(path, dataset_name, 'masks', img0_name_base+'_mask'+image0_name_ext )

    image_0 = read_grimage(img0_filename)[:,:-1]


    im_fft1 = fft2(image_0)
    im_fft_shift1 = np.fft.fftshift(im_fft1)

    #step = 1
    norm = LogNorm(vmin=5, vmax=40000)
    #x = fftfreq(image_0.shape[1], d=step)
    #y = fftfreq(image_0.shape[0], d=step)
    #X, Y = np.meshgrid(x, y)

    axes[0,i].set_title(dataset_name)
    axes[0,i].imshow(image_0,cmap='gray')
    axes[1,i].imshow(np.abs(im_fft_shift1), cmap='viridis',norm=norm)
    #axes[1,0].set_title('Raw')

fig.tight_layout()
fig.subplots_adjust(bottom=0.2, top=0.8, wspace=0.025,hspace=0.025)
