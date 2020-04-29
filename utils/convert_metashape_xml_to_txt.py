#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:48:29 2020

@author: vik748
"""

import xml.etree.ElementTree as ET
import argparse
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser(description='This scripts converts Metashape opencv calib in xml to a yaml file')
parser.add_argument('in_file', type=argparse.FileType('r'), help='file to be converted') #go_pro_icebergs_config.yaml
parser.add_argument('-f', '--factor', type=int, help='factor to divide K matrix', default=1) #go_pro_icebergs_config.yaml

args = parser.parse_args()

calib_xml = args.in_file.name
output_yaml = calib_xml+'.yaml'

# import the same file to read the K and D matrices
tree = ET.parse(calib_xml)
root = tree.getroot()

K_string = root.getchildren()[3].getchildren()[3].text
K = np.fromstring(K_string, dtype=float,sep=" ").reshape((3,3))

K[:2,:] = K[:2,:] / args.factor

D_string = root.getchildren()[4].getchildren()[3].text
D = np.fromstring(D_string, dtype=float,sep=" ")    

with open(output_yaml, 'w') as file:
    documents = yaml.dump({'K': K.tolist(), 'D': D.tolist()}, file)
    