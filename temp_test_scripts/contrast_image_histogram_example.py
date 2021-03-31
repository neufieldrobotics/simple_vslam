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
import progressbar
from datetime import datetime
import matplotlib.ticker as tick


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format

raw_img_folder = '/home/vik748/data/Multibeam_pointcloud_correction/2019_sermilik/paper_images/low_contrast/contrast_examples'

raw_image_names = sorted(glob.glob(raw_img_folder+'/*.png'),reverse=True)

fig1, fig1_axes = plt.subplots(2,len(raw_image_names),figsize=[11.7, 6.2])
hist_y = fig1_axes[1,0]
titles = ['(a) Raw', '(b) 7 bit', '(c) 6 bit', '(d) 5 bit', '(e) 4 bit', '(f) 3 bit']

for i,(name, tit) in enumerate(zip(raw_image_names,titles)):
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    fig1_axes[0,i].imshow(img,cmap='gray',vmin=0, vmax=255)
    fig1_axes[0,i].axis('off')

    fig1_axes[1,i].hist(img.flatten(),256,[0,256])
    fig1_axes[1,i].set_ylim([0,50000])
    fig1_axes[1,i].set_xlim([0,255])
    if i != 0:
        fig1_axes[1,i].axes.get_yaxis().set_ticks([])

    fig1_axes[0,i].set_title(tit)
    fig1_axes[1,i].set_xlabel('Intensity')

hist_y.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
hist_y.set_ylabel('Frequency')
plt.subplots_adjust(wspace=0.05,bottom=0.4,top=0.9, hspace=0.0001)