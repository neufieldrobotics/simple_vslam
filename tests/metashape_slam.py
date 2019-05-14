#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:25:57 2019

@author: vik748
"""
import pickle
import xml.etree.ElementTree as ET


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
#yaml_file = path+'data/Lars2_081018_project_files/data.yml'
pkl_file = path+'data/Lars2_081018_project_files/metashape_export.pkl'
calib_file = path+'data//Lars2_081018_project_files/calibration_export_opencv.xml'

tree = ET.parse(calib_file)
root = tree.getroot()

K_string = root.getchildren()[3].getchildren()[3].text
K = np.fromstring(K_string, dtype=float,sep=" ").reshape((3,3))

D_string = root.getchildren()[4].getchildren()[3].text
D = np.fromstring(D_string, dtype=float,sep=" ")

with open(pkl_file, 'rb') as input:
    frames, points = pickle.load(input)
    

