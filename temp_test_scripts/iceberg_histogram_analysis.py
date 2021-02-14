#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:50:07 2020

@author: vik748
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import sys, os


def plot_img_histograms(gr, axes, title="", mask = None):

    if not mask is None:
        gr[mask==0] = 255
        x_img = gr[mask==1].flatten()
    else:
        x_img = gr.flatten()

    axes[0].imshow(gr,cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(title)

    x = np.linspace(0,255, 256)
    gr_kde_full = st.gaussian_kde(x_img,bw_method='silverman')
    f = gr_kde_full(x)

    # Display Histogram and KDE
    axes[1].hist(x_img, bins=x, color='blue', density=True, alpha=0.4, label='Raw')
    axes[1].fill_between(x, f, color='red',alpha=0.4)
    axes[1].set_xlim(0,255)

    # Display cumulative histogram
    axes[2].hist(x_img, bins=x, color='blue', cumulative=True,
                 density=True, alpha=0.4, label='Raw')

#data_path = '../test_data'
#gr_raw_name = os.path.join(data_path, "museum_raw.png")

if sys.platform == 'darwin':
    data_fold=os.path.expanduser('~/Google Drive/data')
else:
    data_fold=os.path.expanduser('~/data')

gr1_name = os.path.join(data_fold,'Lars1_080818','G0287250.JPG')
gr2_name = os.path.join(data_fold,'Lars2_081018','G0029490.JPG')

gr1_full = cv2.imread(gr1_name, cv2.IMREAD_GRAYSCALE)
gr1 = cv2.resize(gr1_full, (0,0), fx=1/5, fy=1/5, interpolation=cv2.INTER_AREA)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4, 4)
axes = np.empty((3,4),dtype=object)
axes[0,0] = fig.add_subplot(gs[0:2,0])
axes[1,0] = fig.add_subplot(gs[2,0])
axes[2,0] = fig.add_subplot(gs[3,0], sharex = axes[1,0])

axes[0,1] = fig.add_subplot(gs[0:2,1])
axes[1,1] = fig.add_subplot(gs[2,1], sharey = axes[1,0])
axes[2,1] = fig.add_subplot(gs[3,1], sharex = axes[1,1], sharey = axes[2,0])

axes[0,2] = fig.add_subplot(gs[0:2,2])
axes[1,2] = fig.add_subplot(gs[2,2], sharey = axes[1,0])
axes[2,2] = fig.add_subplot(gs[3,2], sharex = axes[1,2], sharey = axes[2,0])

axes[0,3] = fig.add_subplot(gs[0:2,3])
axes[1,3] = fig.add_subplot(gs[2,3], sharey = axes[1,0])
axes[2,3] = fig.add_subplot(gs[3,3], sharex = axes[1,3], sharey = axes[2,0])


[axi.set_axis_off() for axi in axes[0,:].ravel()]
[plt.setp(a.get_xticklabels(), visible=False) for a in axes[1:2,:].ravel()]
[plt.setp(a.get_yticklabels(), visible=False) for a in axes[1:,1:].ravel()]

plot_img_histograms(gr1, axes[:,0], 'Full - Lars 1')

mask_iceberg = np.zeros_like(gr1)
mask_iceberg[205:310,:] = 1
plot_img_histograms(np.copy(gr1), axes[:,1], 'Iceberg', mask = mask_iceberg )

mask_water = np.zeros_like(gr1)
mask_water[320:,:] = 1
plot_img_histograms(np.copy(gr1), axes[:,2], 'Water', mask = mask_water )

mask_sky = np.zeros_like(gr1)
mask_sky[0:190,:] = 1
plot_img_histograms(np.copy(gr1), axes[:,3], 'Sky', mask = mask_sky )


gr2_full = cv2.imread(gr2_name, cv2.IMREAD_GRAYSCALE)
gr2 = cv2.resize(gr2_full, (0,0), fx=1/5, fy=1/5, interpolation=cv2.INTER_AREA)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4, 4)
axes = np.empty((3,4),dtype=object)
axes[0,0] = fig.add_subplot(gs[0:2,0])
axes[1,0] = fig.add_subplot(gs[2,0])
axes[2,0] = fig.add_subplot(gs[3,0], sharex = axes[1,0])

axes[0,1] = fig.add_subplot(gs[0:2,1])
axes[1,1] = fig.add_subplot(gs[2,1], sharey = axes[1,0])
axes[2,1] = fig.add_subplot(gs[3,1], sharex = axes[1,1], sharey = axes[2,0])

axes[0,2] = fig.add_subplot(gs[0:2,2])
axes[1,2] = fig.add_subplot(gs[2,2], sharey = axes[1,0])
axes[2,2] = fig.add_subplot(gs[3,2], sharex = axes[1,2], sharey = axes[2,0])

axes[0,3] = fig.add_subplot(gs[0:2,3])
axes[1,3] = fig.add_subplot(gs[2,3], sharey = axes[1,0])
axes[2,3] = fig.add_subplot(gs[3,3], sharex = axes[1,3], sharey = axes[2,0])


[axi.set_axis_off() for axi in axes[0,:].ravel()]
[plt.setp(a.get_xticklabels(), visible=False) for a in axes[1:2,:].ravel()]
[plt.setp(a.get_yticklabels(), visible=False) for a in axes[1:,1:].ravel()]

plot_img_histograms(gr2, axes[:,0], 'Full - Lars 2')

mask_iceberg = np.zeros_like(gr2)
mask_iceberg[130:280,:] = 1
plot_img_histograms(np.copy(gr2), axes[:,1], 'Iceberg', mask = mask_iceberg )

mask_water = np.zeros_like(gr2)
mask_water[300:600,:] = 1
plot_img_histograms(np.copy(gr2), axes[:,2], 'Water', mask = mask_water )

mask_sky = np.zeros_like(gr2)
mask_sky[0:90,:] = 1
plot_img_histograms(np.copy(gr2), axes[:,3], 'Sky', mask = mask_sky )
