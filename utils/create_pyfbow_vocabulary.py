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
#import data
from zernike.zernike import MultiHarrisZernike
import pyfbow as bow
import time
import glob

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

#data_path = os.path.dirname(os.path.relpath(data.__file__))

if sys.platform == 'darwin':
    data_fold=os.path.expanduser(os.path.join('~','Google Drive','data'))
else:
    data_fold=os.path.expanduser(os.path.join('~','data'))

image_folder = os.path.join(data_fold, 'Stingray2_080718_800x600')

image_names = glob.glob(os.path.join(image_folder,'*.png'))

des_list = []
for image_name in image_names:

    img = cv2.imread(image_name, cv2.IMREAD_COLOR)

    if img is None:
        print ("Couldn't read image: ",image_name)
        pass

    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gr, mask=None)
    print("Detected {} features".format(len(des)))

    des_list.append(des)

st = time.time()
voc.create(des_list[0])

image_folder_name = os.path.split(os.path.dirname(image_names[0]))[-1].replace(' ','_')
detector_config_string = get_detector_config_string(detector)
voc_file_name = image_folder_name + '_' + str(len(image_names)) + '-'+detector_config_string+'.bin'

if os.path.exists(voc_file_name):
    print("Vocaulary file {} exists - Overwriting !!!".format(voc_file_name))

voc.saveToFile(voc_file_name)

print("Created Vocabulary {} in {:.2f} secs".format(voc_file_name,time.time()-st))


