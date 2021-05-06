#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:11:10 2019

@author: vik748
"""
import sys, os
import cv2
import numpy as np
import glob
import yaml
np.set_printoptions(suppress=True)

data_path = os.path.join(os.pardir,'data')
img_folder = 'gopro_chess_board_800x600'

images = sorted(glob.glob(os.path.join(data_path,img_folder,'*.jpg')))

out_corners_file = os.path.join(data_path,img_folder,img_folder+'_corners.yaml')

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

corner_dict = {}

for img_name in images:
    img_basename = os.path.basename(img_name)
    img = cv2.imread(img_name,1)

    while(len(corners)<4):
        #print (len(corners))
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    corners_arr = np.array(corners,dtype=int)
    print(img_basename,": ",corners)
    corner_dict[img_basename] = corners
    corners = []

with open(out_corners_file, 'w') as file:
    _ = yaml.dump(corner_dict, file)

cv2.destroyAllWindows()
