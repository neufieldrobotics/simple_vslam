#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contrast_gradient_histograms

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

def xy_gradients(img):
    '''
    Return x and y gradients of an image. Similar to np.gradient
    '''
    kernelx = 1/2*np.array([[-1,0,1]])
    kernely = 1/2*np.array([[-1],[0],[1]])
    fx = cv2.filter2D(img,cv2.CV_32F,kernelx)
    fy = cv2.filter2D(img,cv2.CV_32F,kernely)
    return fy, fx



img1 = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261.png',1)
img1_clahe = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261_clahe.tif',1)

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr1_clahe = cv2.cvtColor(img1_clahe,cv2.COLOR_BGR2GRAY)

gr1_fy, gr1_fx = xy_gradients(gr1)
gr1_clahe_fy, gr1_clahe_fx = xy_gradients(gr1_clahe)

gr1_fx = np.abs(gr1_fx)
gr1_fy = np.abs(gr1_fy)
gr1_clahe_fx = np.abs(gr1_clahe_fx)
gr1_clahe_fy = np.abs(gr1_clahe_fy)


plt.close('all')
fig1, fig1_axes = plt.subplots(3, 3, num=1, sharey='row')
fig1.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.1, hspace=0.11)
[axi.set_axis_off() for axi in fig1_axes[:2,:].ravel()]

fig1_axes[0,0].imshow(gr1,cmap='gray', vmin=0, vmax=255)
fig1_axes[0,0].set_title("RAW")

fig1_axes[0,1].imshow(gr1_fx,cmap='gray', vmin=0, vmax=64)
fig1_axes[0,1].set_title("Raw fx image")

fig1_axes[0,2].imshow(gr1_fy,cmap='gray', vmin=0, vmax=64)
fig1_axes[0,2].set_title("Raw fy image")

fig1_axes[1,0].imshow(gr1_clahe,cmap='gray', vmin=0, vmax=255)
fig1_axes[1,0].set_title("CLAHE")

fig1_axes[1,1].imshow(gr1_clahe_fx,cmap='gray', vmin=0, vmax=64)
fig1_axes[1,1].set_title("CLAHE fx image")

fig1_axes[1,2].imshow(gr1_clahe_fy,cmap='gray', vmin=0, vmax=64)
fig1_axes[1,2].set_title("CLAHE fy image")


h1,_,_ = fig1_axes[2,0].hist(gr1.flatten(), bins=np.linspace(0,255,256), alpha=0.4, label='Raw')
h1_fx,_,_ = fig1_axes[2,1].hist(gr1_fx.flatten(), bins=np.linspace(0,127,256), alpha=0.4, label='Raw')
h1_fy,_,_ = fig1_axes[2,2].hist(gr1_fy.flatten(), bins=np.linspace(0,127,256), alpha=0.4, label='Raw')

h1_clahe,_,_ = fig1_axes[2,0].hist(gr1_clahe.flatten(), bins=np.linspace(0,255,256), alpha=0.4, label='CLAHE')
h1_clahe_fx,_,_ = fig1_axes[2,1].hist(gr1_clahe_fx.flatten(), bins=np.linspace(0,127,256), alpha=0.4, label='CLAHE')
h1_clahe_fy,_,_ = fig1_axes[2,2].hist(gr1_clahe_fy.flatten(), bins=np.linspace(0,127,256), alpha=0.4, label='CLAHE')

[axi.legend() for axi in fig1_axes[2,:]]

fig2, fig2_axes = plt.subplots(3, 2, num=2, sharey='row')
fig2.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.1, hspace=0.11)
[axi.set_axis_off() for axi in fig2_axes[:2,:].ravel()]

fig2_axes[0,0].imshow(gr1,cmap='gray', vmin=0, vmax=255)
fig2_axes[0,0].set_title("RAW")

fig2_axes[1,0].imshow(gr1_clahe,cmap='gray', vmin=0, vmax=255)
fig2_axes[1,0].set_title("CLAHE")


# Calculate the DoG by subtracting
gr1_dog = cv2.GaussianBlur(gr1,(3,3),0) - cv2.GaussianBlur(gr1,(9,9),0)
gr1_clahe_dog = cv2.GaussianBlur(gr1_clahe,(3,3),0) - cv2.GaussianBlur(gr1_clahe,(9,9),0)

fig2_axes[0,1].imshow(gr1_dog,cmap='gray', vmin=0, vmax=255)
fig2_axes[0,1].set_title("Raw DOG")

fig2_axes[1,1].imshow(gr1_clahe_dog,cmap='gray', vmin=0, vmax=64)
fig2_axes[1,1].set_title("CLAHE DOG image")

h1,_,_ = fig2_axes[2,0].hist(gr1.flatten(), bins=np.linspace(0,255,256), alpha=0.4, label='Raw')
h1_dog,_,_ = fig2_axes[2,1].hist(gr1_dog.flatten(), bins=np.linspace(0,255,256), alpha=0.4, label='Raw')

h1_clahe,_,_ = fig2_axes[2,0].hist(gr1_clahe.flatten(), bins=np.linspace(0,255,256), alpha=0.4, label='CLAHE')
h1_clahe_fx,_,_ = fig2_axes[2,1].hist(gr1_clahe_dog.flatten(), bins=np.linspace(0,255,256), alpha=0.4, label='CLAHE')
[axi.legend() for axi in fig2_axes[2,:]]
fig2_axes[2,1].set_ylim([0,25000])