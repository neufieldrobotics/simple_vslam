#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:52:23 2019

@author: vik748
"""
from zernike import MultiHarrisZernike
import sys
import cv2
import time
from matplotlib import pyplot as plt


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057821.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img2 = cv2.imread(path+'data/skerki_small/all/ESC.970622_023824.0546.tif',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img2 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057826.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

gr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

a = MultiHarrisZernike(Nfeats=50)

import time
st = time.time()
for i in range(1):
    kp, des = a.detectAndCompute(gr1)
print("elapsed: ",(time.time()-st)/1)

outImage	 = cv2.drawKeypoints(gr1, kp, gr1,color=[255,255,0],
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
fig, ax= plt.subplots(dpi=200)
plt.title('Multiscale Harris with Zernike Angles')
plt.axis("off")
plt.imshow(outImage)
plt.show()