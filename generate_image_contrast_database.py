#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import cv2
import sys
import os
import glob
import feat_detector_comparisions_helper# import process_image_contrasts
import progressbar
#from datetime import datetime
import pandas as pd
import ipyparallel as ipp


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('~/data')
import time

#raw_sets_folder = 'Lars2_081018_800x600'
raw_sets_folder = 'Stingray2_080718_800x600'

raw_img_folder = os.path.join(path,raw_sets_folder)
mask_folder = raw_img_folder.replace('_800x600', '_masks_from_model_800x600')
ctrst_img_output_folder = raw_img_folder + "_ctrst_imgs"
if not os.path.exists(ctrst_img_output_folder):
    os.makedirs(ctrst_img_output_folder)


raw_image_names = sorted(glob.glob(raw_img_folder+'/*.png'))


base_settings = {'set_title': os.path.basename(raw_img_folder)}

image_skip = 5

contrast_adj_factors = np.arange(0,-1.1,-.1)  #np.array([0.0, -0.5, -1.0])

image_names = raw_image_names[0:25:image_skip]

#img_name = image_names[0]

st = time.time()
#img_df = process_image(img_name, contrast_adj_factors=contrast_adj_factors, mask_folder=mask_folder,
#                       ctrst_img_output_folder = ctrst_img_output_folder)

ipp_client = ipp.Client()
ipp_dview = ipp_client[:]
curr_dir = os.path.dirname(__file__)
print(curr_dir)
ipp_dview.execute("import sys,os; \
                   from feat_detector_comparisions_helper import process_image_contrasts".format(curr_dir, curr_dir))
#print("executed")
ipp_dview.push({'contrast_adj_factors': contrast_adj_factors, 'base_settings': base_settings,
                'mask_folder': mask_folder, 'ctrst_img_output_folder': ctrst_img_output_folder})

process_img_lam = lambda img_name:process_image_contrasts(img_name, contrast_adj_factors=contrast_adj_factors, mask_folder=mask_folder,
                                                          ctrst_img_output_folder = ctrst_img_output_folder, base_settings=base_settings)

img_df_list = ipp_dview.map_sync(process_img_lam, image_names)
#img_df_list = list(map(process_img_lam, image_names))
elapsed_time = time.time()-st


print("Images processed in : {:.2f}. Per image: {:.2f} ".format(elapsed_time, elapsed_time / len(image_names)))

img_df=pd.concat(img_df_list,ignore_index=True)
output_file = os.path.join(os.path.abspath(os.path.join(raw_img_folder, os.pardir)),
                           "Contrast_images_df_"+raw_sets_folder+time.strftime("_%Y%m%d_%H%M%S")+".csv")
img_df.to_csv(output_file)
print("Results written to {}".format(output_file))