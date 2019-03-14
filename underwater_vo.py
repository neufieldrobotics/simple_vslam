#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from vslam_helper import *
import yaml
import glob
import re
import argparse
np.set_printoptions(precision=3,suppress=True)
from scipy.io import loadmat

print (sys.platform)

# Inputs, images and camera info

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
    window_xadj = 0
    window_yadj = 45
else:
    path = '/home/vik748/'
    window_xadj = 65
    window_yadj = 430
    
mat_folder = 'data/tape_mats'

mats = sorted([f for f in glob.glob(path+mat_folder+'/match*') 
                 if re.match('^.*\.'+'mat'+'$', f, flags=re.IGNORECASE)])
mat_names = [m.split('/match')[-1].split('.mat')[0] for m in mats]

all_pos = []    
for m in mats:
    pos = m.split('/match')[-1].split('.mat')[0].split('_')
    if not pos[0] in all_pos:
        all_pos.append(pos[0])
    if not pos[1] in all_pos:
        all_pos.append(pos[1])

all_pos = sorted(all_pos)
links = np.zeros((len(all_pos),len(all_pos)),dtype=bool)
for i,pos1 in enumerate(all_pos):
    for j,pos2 in enumerate(all_pos):
        file_string = pos1+'_'+pos2
        if (file_string in mat_names): links[i,j]=True
        
for pos1,pos2 in zip(all_pos[:-1],all_pos[1:]):
    mat_file = loadmat(mats[mat_names.index(pos1+"_"+pos2)])
    lt = mat_file['link_type'][0,0]
    ff = mat_file['ff']
    gg = mat_file['gg']
    
    retval, mask	=	cv2.findHomography(	ff, gg,  cv2.RANSAC)
    print(retval)
