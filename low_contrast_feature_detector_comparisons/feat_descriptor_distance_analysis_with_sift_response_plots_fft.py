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

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

dataset_name = datasets[2]
contrast_type = contrast_types[0]

img0_name = image_names[dataset_name]
img0_name_base, image0_name_ext = os.path.splitext(img0_name)
img0_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
img0_mask_filename = os.path.join(path, dataset_name, 'masks', img0_name_base+'_mask'+image0_name_ext )

image_0 = read_grimage(img0_filename)[:,:-1]

dataset_name = datasets[3]
contrast_type = contrast_types[0]

img1_name = image_names[dataset_name]
img1_name_base, image0_name_ext = os.path.splitext(img0_name)
img1_filename = os.path.join(path, dataset_name, dataset_name+'_'+contrast_type, img0_name )
img1_mask_filename = os.path.join(path, dataset_name, 'masks', img0_name_base+'_mask'+image0_name_ext )

image_1 = read_grimage(img1_filename)[:,:-1]

l1 = image_0[round(image_0.shape[0]/2)]
l2 = image_1[round(image_1.shape[0]/2)]

n = l1.size
step = 1
freq = fftfreq(n, d=step)

yf1 = fft(l1)
yf2 = fft(l2)

fig1, ax = plt.subplots(1,1)
ax.plot(freq,np.abs(yf1),'.',label='Raw')
ax.plot(freq,np.abs(yf2),'.',label='CLAHE')
ax.legend()
ax.set_yscale('log')
ax.set_ylim([1,None])
ax.set_xlabel('Fourier Frequency')
ax.set_ylabel('Fourier Amplitude')

#plt.gca().set_ylim([0, 2500])

im_fft1 = fft2(image_0)
im_fft2 = fft2(image_1)
im_fft_shift1 = np.fft.fftshift(im_fft1)
im_fft_shift2 = np.fft.fftshift(im_fft2)

fig, axes = plt.subplots(2,2)
norm = LogNorm(vmin=5, vmax=20000)
x = fftfreq(image_0.shape[1], d=step)
y = fftfreq(image_0.shape[0], d=step)
X, Y = np.meshgrid(x, y)

[ax.axis('off') for ax in axes.ravel()]

axes[0,0].imshow(image_0,cmap='gray')
axes[0,1].imshow(image_1,cmap='gray')

axes[1,0].imshow(np.abs(im_fft_shift1), cmap='viridis',norm=norm)
#axes[1,0].set_title('Raw')
axes[1,1].imshow(np.abs(im_fft_shift2), cmap='viridis',norm=norm)
#axes[1,1].set_title('CLAHE')