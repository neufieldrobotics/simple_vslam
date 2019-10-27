#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:47:19 2019

@author: vik748
"""
import numpy as np
import cv2
import glob
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from matplotlib import pyplot as plt
np.set_printoptions(precision=8,suppress=False)
from matlab_imresize.imresize import imresize
import progressbar

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
fold = "icebergs/animation_other/"
images = sorted(glob.glob(path+fold+'*.png'))
outfold = "icebergs/animation_other_1920x1080/"

os.makedirs(path+outfold, exist_ok=True)

fig = plt.figure(1)

for fname in progressbar.progressbar(images,redirect_stdout=True):
    print ("File - ", fname)
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    #imgrs = cv2.resize(img, (800,600), interpolation=cv2.INTER_LANCZOS4)
    
    #imgrs = imresize(img, scalar_scale=1/5, method='bicubic')
    imgrs = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_CUBIC)

           
    outname = path+outfold+fname.split('/')[-1][:-3] + 'png'
    
    retval = cv2.imwrite(outname, imgrs)
    
    print (retval, " Writing to - ", outname)
    #plt.imshow(imgrs)
    #plt.draw()
    #plt.pause(0.01)

plt.close(fig='all')