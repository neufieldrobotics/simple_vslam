#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:19:22 2020

@author: vik748
"""
import os, sys
if os.path.dirname(os.path.realpath(__file__)) == os.getcwd():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cv2
import data
from zernike.zernike import MultiHarrisZernike
import pyfbow as bow
import time
import glob
import numpy as np

k = 10
L = 6
nthreads = 1
maxIters = 0
verbose = False

def get_detector_config_string(detector):
    '''
    Get unique config string for detector config
    '''
    get_func_names = sorted([att for att in detector.__dir__() if att.startswith('get')])
    get_vals = [getattr(detector, func_name)() for func_name in get_func_names]
    get_vals_str = [str(val) if type(val)!=float else "{:.2f}".format(val) for val in get_vals]
    return '_'.join(get_vals_str)

voc = bow.Vocabulary(k, L, nthreads, maxIters, verbose)

zernike_detector = MultiHarrisZernike(Nfeats= 600, seci = 5, secj = 4, levels = 6,
                                      ratio = 0.75, sigi = 2.75, sigd = 1.0, nmax = 8,
                                      like_matlab=False, lmax_nd = 3, harris_threshold = None)

orb_detector = cv2.ORB_create(nfeatures=1200, WTA_K= 2, edgeThreshold= 31, patchSize= 31,
                              fastThreshold= 3, firstLevel= 0, nlevels= 6, scaleFactor= 1.2,
                              scoreType= 0)

detector = orb_detector

data_path = os.path.dirname(os.path.relpath(data.__file__))

#image_names = glob.glob(os.path.join(data_path,'contrast_test_images')+'/*')

image_names = glob.glob('/Users/vik748/Google Drive/data/Stingray2_080718_800x600/*.png')

voc.readFromFile('Stingray2_080718_800x600_1480-MultiHarrisZernike_600_0_6_3_8_0.75_5_4_1.00_2.75.bin')


des_list = []
for image_name in image_names[::10]:

    img = cv2.imread(image_name, cv2.IMREAD_COLOR)

    if img is None:
        print ("Couldn't read image: ",image_name)
        pass

    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gr, mask=None)
    print("Detected {} features".format(len(des)))

    des_list.append(des)

query_des = des_list[int(len(des_list)/2)]
query_fbow = voc.transform(query_des)
scores = np.zeros(len(des_list))
for i,des in enumerate(des_list):

    scores[i] = bow.fBow.score(voc.transform(des), query_fbow)
