#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:11:10 2019

@author: vik748
"""
import sys
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
#%matplotlib inline
np.set_printoptions(suppress=True)

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
    
images = sorted(glob.glob(path+'data/chess_board3/*.JPG'))

def draw_circle(event,x,y,flags,param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),10,(0,255,0),-1)
        corners.append([int(x),int(y)])
        #print (corners)

corners = []
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',(1200,900))
cv2.setMouseCallback('image',draw_circle)

for img_name in images:
    #img_name = images[0]
    img = cv2.imread(img_name,1)
    
    while(len(corners)<4):
        #print (len(corners))
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    
    corners_arr = np.array(corners,dtype=int)
    print(img_name.split('/')[-1],": ",corners)
    corners = []
        
#cv2.destroyAllWindows()
