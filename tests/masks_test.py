#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:14:47 2019

@author: vik748
"""
import numpy as np
import cv2

def tiled_features_detect(detector, img, tiley, tilex, border):
    def draw_mask(xstart,ystart,height,width,W,H):
        mask = np.zeros((H,W), dtype=np.uint8)
        mask[xstart:xstart+height,ystart:ystart+width] = 1
        return mask
    HEIGHT, WIDTH = img.shape
    assert WIDTH%tiley == 0, "Width is not a multiple of tilex"
    assert HEIGHT%tilex == 0, "Height is not a multiple of tiley"
    w_width = int(WIDTH/tiley)
    w_height = int(HEIGHT/tilex)
    
    #mask = np.zeros(size, dtype=np.uint8)
    
    xx = np.linspace(0,HEIGHT-w_height,tilex,dtype='int')
    yy = np.linspace(0,WIDTH-w_width,tiley,dtype='int')
        
    kps = []
    dess = []
    
    for ix in xx:
        for iy in yy:
            if ix == 0: 
                start_x = ix
                win_h = w_height
            else: 
                start_x = ix - border
                win_h = w_height+border
                
            if iy == 0:
                start_y = iy
                win_w = w_width
            else:
                start_y = iy - border 
                win_w = w_height+border
                
            #print ('Mask starting at: ',start_x,', ',win_h,'high , ',start_y,',',win_w,' wide')
            m = draw_mask(start_x,start_y,win_h,win_w,WIDTH,HEIGHT)
            kp,des = detector.detectAndCompute(img,m)
            kps.extend(kp)
            dess.extend(des)
    return kps, dess




size = (4000,3000)
WIDTH, HEIGHT = size
tiley = 8
tilex = 6
border = 128

            #print(m)
            #input ('space')
            #cv2.imshow('image', m*255)
            #cv2.waitKey()