#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:58:28 2019

@author: vik748
"""
import numpy as np
from matplotlib import pyplot as plt

name = 'results_array_baseline_20_20191105175448.csv'
results_array = np.genfromtxt('/Users/vik748/Google Drive/data/feature_descriptor_comparision/results/clahe_images/'+name, delimiter=',')

plt.figure(3); plt.cla()
bins = np.linspace(10, 250, 25)

plt.hist(results_array, bins=bins, alpha=0.5, label=['Zernike','ORB','SIFT'])
plt.suptitle("Lars 1 800x600 CLAHE - Baseline - 20 secs apart")
plt.legend(loc='upper right')
#plt.axes().set_ylim([0, 750])
plt.xlabel('Bins (Number of matches)')
plt.ylabel('Occurances (Image pairs)')
plt.show()