#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 08:40:22 2019

@author: vik748
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from vslam_helper import *

#pose_dict = read_metashape_poses('/Users/vik748/Google Drive/data/Multibeam_pointcloud_correction/Lars2_081018/Lars_2_081018_camera_poses_mts_20200428.txt')
pose_dict = read_metashape_poses('/home/vik748/data/Multibeam_pointcloud_correction/Lars2_081018/Lars_2_081018_camera_poses_mts_20200428.txt')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')


for img in sorted(pose_dict)[::20]:
    plot_pose3_on_axes(ax,pose_dict[img], axis_length=5)

set_axes_equal(ax)

plt.show()
