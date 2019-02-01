#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 08:40:22 2019

@author: vik748
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vslam_helper import *


from vslam_helper import *

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')

plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=0.5)

R = np.array([[ 0.751, -0.075, -0.656],
              [ 0.023,  0.996, -0.087],
              [ 0.66 ,  0.05 ,  0.749]])

t = np.array([[0.86 ],[0.134],[0.492]])

R_inv, t_inv = pose_inv(R, t)


plot_pose3_on_axes(ax,R.T , np.zeros(3)[np.newaxis], axis_length=1.0)

set_axes_equal(ax)

plt.show()