#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:25:57 2019

@author: vik748
"""
import pickle

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
yaml_file = path+'data/Lars2_081018_project_files/data.yml'
pkl_file = path+'data/Lars2_081018_project_files/metashape_export.pkl'

#with open(yaml_file, 'r') as stream:
#    frames = yaml.safe_load(stream)
    
with open(pkl_file, 'rb') as input:
    frames = pickle.load(input)