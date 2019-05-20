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
np.set_printoptions(precision=8,suppress=False)
from matlab_imresize.imresize import imresize
import progressbar
import exifread
import utm

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
fold = "data/Lars2_081018/"
images = sorted(glob.glob(path+fold+'*.JPG'))
#outfold = "data/Lars2_081018_800x600/"

os.makedirs(path+outfold, exist_ok=True)

fig = plt.figure(1)

def val2float(value):
    return float(value.num) / float(value.den)

def _convert_to_degress(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = val2float(value.values[0])
    m = val2float(value.values[1])
    s = val2float(value.values[2])

    return d + (m / 60.0) + (s / 3600.0)

x_list = []
y_list = []
z_list = []
for fname in progressbar.progressbar(images,redirect_stdout=True):
    with open(fname, 'rb') as f:
        tags = exifread.process_file(f)
        latitude = tags.get('GPS GPSLatitude')
        latitude_ref = tags.get('GPS GPSLatitudeRef')
        longitude = tags.get('GPS GPSLongitude')
        longitude_ref = tags.get('GPS GPSLongitudeRef')
        altitude = tags.get('GPS GPSAltitude')
        altitude_ref = tags.get('GPS GPSAltitudeRef')
        if latitude:
            lat_value = _convert_to_degress(latitude)
            if latitude_ref.values != 'N':
                lat_value = -lat_value
        else:
            continue
        if longitude:
            lon_value = _convert_to_degress(longitude)
            if longitude_ref.values != 'E':
                lon_value = -lon_value
        else:
            continue
        #print( {'latitude': lat_value, 'longitude': lon_value})
        x,y,zone_num,zone_letter =  utm.from_latlon(lat_value, lon_value)
        x_list.append(x)
        y_list.append(y)
        z_list.append(val2float(altitude.values[0]))

plt.plot(x_list, y_list)
