#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import sys
from matplotlib import pyplot as plt
import os
import pandas as pd
import scipy.stats as st
from feat_detector_comparisions_helper import *
sys.path.insert(0, os.path.abspath('..'))
import re

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('~/data/')

results_file = 'results/matching_results_20200710_092947_Lars1_full.csv'
#results_file = 'results/matching_results_20200717_180713_Stingray2.csv'
#results_file = 'results/matching_results_20200711_053758_Lars2_full.csv'
#results_file = 'results/matching_results_20200711_053758_Lars2_full.csv'
#results_file = 'results/matching_results_20200720_143024_Stingray2_with_mask_names_fixed.csv'
#results_file = 'results/matching_results_20200722_003934-Morgan1_072719_800x600.csv'
#results_file = 'results/matching_results_20200727_203856_Stingray2_080718_800x600_KLT_baseline_20_threshold_0.0001.csv'
#results_file = 'results/matching_results_20200730_081323_Stingray2_4bit_threshold_0.001.csv'
#results_file = 'results/matching_results_20200731_003948_Stingray2_080718_800x600_3bit_threshold_0.0001.csv'
#results_file = 'results/matching_results_20200731_012906_matching_results_Stingray2_080718_800x600_KLT_3bit_threshold_0.001.csv'
#results_file = 'results/matching_results_20200728_002757_Morgan1_072719_800x600_KLT_baseline_20_threshold_0.001.csv'
#results_file = 'results/matching_results_20200727_221010_Morgan1_072719_800x600_KLT_baseline_20_threshold_0.0001.csv'
#results_file = 'results/matching_results_20200728_001519_Morgan2_073019_800x600_KLT_baseline_20_threshold_0.001.csv'
#results_file = 'results/matching_results_20200727_232430_Morgan2_073019_800x600_KLT_baseline_20_threshold_0.0001.csv'
#results_file = max(glob.glob('results/matching*.csv'), key=os.path.getctime)

results_file = '/home/vik748/data/low_contrast_results/matching_results/matching_results_20200710_092947_Lars1_full.csv'

#threshold = float(re.findall("threshold_[0-9]*([.][0-9]+)", results_file)[0])

results_df = pd.read_csv(results_file)
#results_df2 = pd.read_csv(results_file2)
#results_df3 = pd.read_csv(results_file3)

#results_df = pd.concat([results_df1, results_df2, results_df3], ignore_index=True)

#results_df.loc[(results_df['detector'] == 'ORB') && (results_df['descriptor'] == 'ORB')]
contrast_adj_factors = np.arange(0,-1.1,-.1)

fig,axes = plt.subplots(1,3, sharex=True, sharey=True,num=1, figsize=[8.5, 4.  ])
#ax = axes[1]
detdes_dict_list = [{'detector': 'ORB', 'descriptor': 'ORB'},
                    {'detector': 'MultiHarrisZernike', 'descriptor': 'MultiHarrisZernike'},
                    {'detector': 'MultiHarrisZernike', 'descriptor': 'ORB'}]

baselines = [5,10,20]
display_names = ['ORB', 'Harris-Zernike', 'Harris-rBRIEF']

#detdes_dict_list = [{'detector': 'MultiHarrisZernike', 'descriptor': 'ORB'}]

for ax,bl in zip(axes,baselines):
    for detdes_dict,dn in zip(detdes_dict_list,display_names):
    
        plot_dict = { **detdes_dict,
                     'set_title':'Lars1_080818_800x600', #Morgan1_072719_800x600
                     'contrast_adj_factor': 0.0,
                     'baseline': bl}
        print(detdes_dict,dn)
    
        bool_series = pd.Series(index=results_df.index, dtype=bool,data=1)
    
        for it, val in plot_dict.items():
            bool_series = bool_series & ( results_df[it] == val )
    
        relevant_matches = results_df[bool_series]['matches'].values
    
        matches_kde_full = st.gaussian_kde(relevant_matches, bw_method='scott')
        x = np.linspace(0,225,226)
        matches_x = matches_kde_full(x)
        ax.plot(x, matches_x, label=dn)

    ax.set_title("Baseline: {}".format(bl)) #threshold:{:.4f} ,threshold
    ax.set_xlabel('Number of matches per pair')
    #ax.legend()
    ax.set_xlim([0,None])
    ax.set_ylim([0,0.015])

fig.subplots_adjust(bottom=0.2, top=0.8, wspace=0.05)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,loc="lower center", ncol=3)

axes[0].set_ylabel('Gaussian Kernel Denisty Estimates')
#fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.05, hspace=0.05)
fig.suptitle("Dataset: {}".format(plot_dict['set_title']))
plt.savefig('pairwise_results.pdf',format='pdf', dpi=300)
