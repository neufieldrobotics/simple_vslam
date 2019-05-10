#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:28:59 2019

@author: vik748
"""
def read_cam_lines(f, N):
    lines = [line for line in [f.readline().replace('\n',' ') for _ in range(N)] if len(line) ]
    R = np.fromstring(''.join(lines[1:4]),dtype=float,sep=" ").reshape((3,3))
    t = np.fromstring(lines[4],dtype=float, sep=" ").reshape((3,1))
    return np.hstack((R,t)),t.T
    

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
file = path+'data/Lars2_081018_project_files/camera_export.out'

f = open(file,'r')
if not f.readline().startswith("# Bundle file v0.3"):
    raise ValueError ("File doesnt start with \"# Bundle file v0.3\" ")

[num_cameras, num_points] = [int(x) for x in f.readline().strip().split()]

pos = np.empty((0,3))
for i in range(num_cameras):
    [R, t] = read_cam_lines(f,5)
    pos = np.vstack((pos,t))
