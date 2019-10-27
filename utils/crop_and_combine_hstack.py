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
#from matlab_imresize.imresize import imresize
import progressbar

if sys.platform == 'darwin':
    path = '/Users/vik748/'
else:
    path = '/home/vik748/'
fold = "corrected_frames/"
ren_image_folder = "uncorrected_frames/"

raw_images = sorted(glob.glob(path+fold+'*.png'))

outfold = "combined/"

os.makedirs(path+outfold, exist_ok=True)

fig = plt.figure(1)

aspect_ratio = 16 / 9
img = cv2.imread(raw_images[0], cv2.IMREAD_COLOR)
img_width = img.shape[1]
req_img_height = int(img_width / aspect_ratio / 2)
top_crop = 650


for fname in progressbar.progressbar(raw_images,redirect_stdout=True):
    print ("File - ", fname)
    #ren_filename_base, ren_filename_ext = os.path.splitext(os.path.basename(fname))
    #ren_filename = ren_filename_base + '_render' + '.jpg'
    #ren_fullname = os.path.join(ren_image_folder,ren_filename)
    
    raw_img = cv2.imread(fname, cv2.IMREAD_COLOR)
    ren_img = cv2.imread(path+ren_image_folder+os.path.basename(fname), cv2.IMREAD_COLOR)
    
    #raw_img_crop = raw_img[top_crop : top_crop + req_img_height,:]
    #ren_img_crop = ren_img[top_crop : top_crop + req_img_height,:]
    
    #out_img = np.hstack((ren_img_crop[:,100:-100], raw_img[:,100:-100]))
    out_img = np.hstack((ren_img[:,120:-160], raw_img[:,120:-160]))
    
    out_img_rs = cv2.resize(out_img, (1920,1080), interpolation = cv2.INTER_CUBIC)
               
    outname = path+outfold+fname.split('/')[-1]
    
    retval = cv2.imwrite(outname, out_img_rs)
    #print (retval, " Writing to - ", outname)
    #plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    #plt.draw()
    #plt.pause(0.1)

plt.close(fig='all')