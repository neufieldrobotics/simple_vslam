#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:47:19 2019

@author: vik748
"""
import numpy as np
import cv2
import glob
import sys
from matplotlib import pyplot as plt
np.set_printoptions(precision=8,suppress=False)


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
fold = "data/time_lapse_5_cervino_masks_out/"
images = sorted(glob.glob(path+fold+'*.png'))
outfold = "data/time_lapse_5_cervino_800x600_masks_out/"

fig = plt.figure(1)

for fname in images:
    print ("File - ", fname)
    img = cv2.imread(fname)
    imgrs = cv2.resize(img, (800,600), interpolation=cv2.INTER_LANCZOS4)
    
       
    outname = path+outfold+fname.split('/')[-1][:-3] + 'png'
    
    retval = cv2.imwrite(outname, imgrs)
    print (retval, " Writing to - ", outname)
    #plt.imshow(imgrs)
    #plt.draw()
    #plt.pause(0.01)


plt.close(fig='all')