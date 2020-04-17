#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:25:57 2019

@author: vik748
"""
import pickle
import sys,os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from vslam_helper import *

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
#yaml_file = path+'data/Lars2_081018_project_files/data.yml'
#pkl_file = path+'data/Lars2_081018_project_files/metashape_export.pkl'
#pkl_file = path+'data/Metashape_Bundler_export/Lars1_08082018_metashape_pts_and_cams_gs_750_75_valid_only_plus_five_invalid.pkl'
pkl_file = path+'data/Metashape_Bundler_export/Stingray2_08072018_metashape_pts_and_cams.pkl'
#calib_file = path+'data//Lars2_081018_project_files/calibration_export_opencv.xml'

with open(pkl_file, 'rb') as input:
    frames, points,K,D = pickle.load(input)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
fig2.subplots_adjust(0,0,1,1)
#plt.get_current_fig_manager().window.setGeometry(1036+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
ax2.set_aspect('equal')         # important!
ax2.view_init(0, 90)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
 
for i,frame in enumerate(frames):
    if i%10==0:
        plot_pose3_on_axes(ax2, frame['transform'], axis_length=1.0)
        plt.pause(.1)
        #set_axes_equal(ax2)
