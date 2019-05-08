
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
np.set_printoptions(precision=4,suppress=True)
from matlab_imresize.imresize import imresize
import pickle


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
file = open(path+"data/kitti/00/00.txt","r")
out = path+"data/kitti/00/00.pkl"
#os.makedirs(path+outfold, exist_ok=True)
ground_truth_poses = {}
ground_truth_trail = np.empty([0,3])

for i,line in enumerate(file):
    arr = np.array([float(a) for a in (line.split(' '))])
    R = arr.reshape([3,4])
    T = np.vstack((R,np.array([[0,0,0,1]])))
    ground_truth_poses[i] = T
    ground_truth_trail = np.append(ground_truth_trail, T[:3,[-1]].T,axis=0)

with open(out, 'wb') as output:
            pickle.dump(ground_truth_poses, output)
            
plt.close(fig='all')