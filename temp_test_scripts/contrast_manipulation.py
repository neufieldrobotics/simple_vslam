#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:39:12 2020

@author: vik748
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_Pose_test

@author: vik748
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt, colors as clrs
from mpl_toolkits.mplot3d import Axes3D
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from vslam_helper import *
from matlab_imresize.imresize import imresize
from zernike.zernike import MultiHarrisZernike


def adj_contrast(gr, alpha, beta=0):
    '''
    Adjust contrast of the image using slope alpha and brightness beta
    where f(x)=alpha((x−128) + 128 + beta
    '''
    assert gr.ndim == 2, "Number of image dims != 2, possibly rgb image"
    return (np.round(alpha*(gr)+0+beta).astype(np.uint8) )

def naive_contast_enhancement(gr):
    '''
    '''
    gr1 = gr - np.min(gr)
    gr2 = gr1 * 255.0 / np.max(gr1)
    return np.round(gr2).astype(np.uint8)

def cdf_hist_eq(gr):
    # Flatten the image into 1 dimension: pixels
    pixels = gr.flatten()

    # Generate a cumulative histogram
    cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
    new_pixels = np.interp(pixels, bins[:-1], cdf*255)

    # Reshape new_pixels as a 2-D array: new_image
    gr_eq = new_pixels.reshape(gr.shape)
    return np.round(gr_eq).astype(np.uint8)


img1 = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261.png',1)
#img1 = img1[:300,:]

img1_clahe = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261_clahe.tif',1)
#img1_clahe = img1_clahe[:300,:]

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr1_clahe = cv2.cvtColor(img1_clahe,cv2.COLOR_BGR2GRAY)

plt.close('all')
fig1, fig1_axes = plt.subplots(2, 4, num=1, sharey='row')
fig1.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.1, hspace=0.0)
fig1_axes[0,0].axis("off")
fig1_axes[0,1].axis("off")
fig1_axes[0,2].axis("off")
fig1_axes[0,3].axis("off")


fig1_axes[0,0].imshow(gr1,cmap='gray', vmin=0, vmax=255)
fig1_axes[0,0].set_title("RAW")
fig1_axes[0,1].imshow(gr1_clahe,cmap='gray', vmin=0, vmax=255)
fig1_axes[0,1].set_title("CLAHE")
#fig2, fig2_axes = plt.subplots(2, 2, num=2, sharey='row')
#fig2.subplots_adjust(left=0.05, bottom=0.1, right=1.0, top=.9, wspace=0.1, hspace=0.0)
#fig2_axes[0,1].axis("off")

h1,_,_ = fig1_axes[1,0].hist(gr1.flatten(), bins=np.linspace(0,255,256), alpha=0.5)
h1_clahe,_,_ = fig1_axes[1,1].hist(gr1_clahe.flatten(), bins=np.linspace(0,255,256), alpha=0.5)
fig1_axes[1,0].set_yticklabels(['{:,}'.format(x / 1000) for x in fig1_axes[1,0].get_yticks()])


gr1_clahe_adj = adj_contrast(gr1_clahe, alpha=1/16, beta=0)
fig1_axes[0,2].set_title("CLAHE - red contrast")
fig1_axes[0,2].imshow(gr1_clahe_adj,cmap='gray', vmin=0, vmax=255)
h1_clahe_adj,_,_ = fig1_axes[1,2].hist(gr1_clahe_adj.flatten(), bins=np.linspace(0,255,256), alpha=0.5)



gr1_clahe_adj = naive_contast_enhancement(img1)
fig1_axes[0,3].set_title("CLAHE, red and then Naive")

'''
clahe = cv2.createCLAHE(clipLimit=20., tileGridSize=(16,12))
gr1_clahe_adj = clahe.apply(gr1_clahe_adj)
fig1_axes[0,3].set_title("CLAHE, red and then CLAHE")

gr1_clahe_adj = cdf_hist_eq(gr1_clahe_adj)
fig1_axes[0,3].set_title("CLAHE, red and then CDF Eq")
'''

fig1_axes[0,3].imshow(gr1_clahe_adj,cmap='gray', vmin=0, vmax=255)
h1_clahe_adj,_,_ = fig1_axes[1,3].hist(gr1_clahe_adj.flatten(), bins=np.linspace(0,255,256), alpha=0.5)



