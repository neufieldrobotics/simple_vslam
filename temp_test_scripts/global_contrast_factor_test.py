#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:35:55 2020

@author: vik748
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_image_average_contrast(k, gamma=2.2):
    L = 100 * np.sqrt((k / 255) ** gamma )
    # pad image with border replicating edge values
    L_pad = np.pad(L,1,mode='edge')

    # compute differences in all directions
    left_diff = L - L_pad[1:-1,:-2]
    right_diff = L - L_pad[1:-1,2:]
    up_diff = L - L_pad[:-2,1:-1]
    down_diff = L - L_pad[2:,1:-1]

    # create matrix with number of valid values 2 in corners, 3 along edges and 4 in the center
    num_valid_vals = 3 * np.ones_like(L)
    num_valid_vals[[0,0,-1,-1],[0,-1,0,-1]] = 2
    num_valid_vals[1:-1,1:-1] = 4

    pixel_avgs = (np.abs(left_diff) + np.abs(right_diff) + np.abs(up_diff) + np.abs(down_diff)) / num_valid_vals

    return np.mean(pixel_avgs)

def compute_global_contrast_factor(img):
    if img.ndim != 2:
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gr = img

    superpixel_sizes = [1, 2, 4, 8, 16, 25, 50, 100, 200]

    gcf = 0

    for i,size in enumerate(superpixel_sizes,1):
        wi =(-0.406385 * i / 9 + 0.334573) * i/9 + 0.0877526
        im_scale = cv2.resize(gr, (0,0), fx=1/size, fy=1/size,
                              interpolation=cv2.INTER_LINEAR)
        avg_contrast_scale = compute_image_average_contrast(im_scale)
        gcf += wi * avg_contrast_scale

    return gcf


img1 = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261.png',1)
img1_clahe = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261_clahe.tif',1)

img1_gcf = compute_global_contrast_factor(img1)
img1_clahe_gcf = compute_global_contrast_factor(img1_clahe)

img2 = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0029482.png',1)
img2_clahe = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0029482_clahe.tif',1)

img2_gcf = compute_global_contrast_factor(img2)
img2_clahe_gcf = compute_global_contrast_factor(img2_clahe)


fig1, fig1_axes = plt.subplots(2, 2, num=1)
fig1.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.1, hspace=0.11)
[axi.set_axis_off() for axi in fig1_axes[:2,:].ravel()]

fig1_axes[0,0].imshow(img1,cmap='gray', vmin=0, vmax=255)
fig1_axes[0,0].set_title('Img1 Raw\nGCF={:.2f}'.format(img1_gcf))
fig1_axes[0,1].imshow(img1_clahe,cmap='gray', vmin=0, vmax=255)
fig1_axes[0,1].set_title('Img1 CLAHE\nGCF={:.2f}'.format(img1_clahe_gcf))

fig1_axes[1,0].imshow(img2,cmap='gray', vmin=0, vmax=255)
fig1_axes[1,0].set_title('Img2 Raw\nGCF={:.2f}'.format(img2_gcf))
fig1_axes[1,1].imshow(img2_clahe,cmap='gray', vmin=0, vmax=255)
fig1_axes[1,1].set_title('Img2 CLAHE\nGCF={:.2f}'.format(img2_clahe_gcf))

