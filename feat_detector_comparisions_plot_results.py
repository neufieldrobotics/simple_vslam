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

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('~/data/')

results_file = 'results/matching_results_20200709_225904_combined.csv'

results_df = pd.read_csv(results_file)
#results_df2 = pd.read_csv(results_file2)
#results_df3 = pd.read_csv(results_file3)

#results_df = pd.concat([results_df1, results_df2, results_df3], ignore_index=True)

#results_df.loc[(results_df['detector'] == 'ORB') && (results_df['descriptor'] == 'ORB')]
contrast_adj_factors = np.arange(0,-1.1,-.2)

fig,axes = plt.subplots(1,3, sharex=True, sharey=True)

detdes_dict_list = [{'detector': 'ORB', 'descriptor': 'ORB'},
                    {'detector': 'MultiHarrisZernike', 'descriptor': 'ORB'},
                    {'detector': 'MultiHarrisZernike', 'descriptor': 'MultiHarrisZernike'}]

for detdes_dict, ax in zip(detdes_dict_list, axes):
    
    for cont_fact in contrast_adj_factors:    
        plot_dict = { **detdes_dict,
                     'set_title':'Lars1_080818_800x600',
                     'contrast_adj_factor': cont_fact,
                     'baseline': 20}
         
        bool_series = pd.Series(index=results_df.index, dtype=bool,data=1)
        
        for it, val in plot_dict.items():
            bool_series = bool_series & ( results_df[it] == val )
        
        relevant_matches = results_df[bool_series]['matches'].values
        
        matches_kde_full = st.gaussian_kde(relevant_matches, bw_method='scott')
        x = np.linspace(0,200,201)
        matches_x = matches_kde_full(x)
        ax.plot(x, matches_x, label='ctrst fact={:.1f}'.format(cont_fact))

    ax.set_title("Det: {} Des: {}".format(plot_dict['detector'],plot_dict['descriptor']))
    ax.set_xlabel('Number of matches per pair')
    ax.legend()
    
axes[0].set_ylabel('Gaussian Kernel Denisty Estimates')
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.05, hspace=0.05)
fig.suptitle("Dataset: {}, baseline: {}".format(plot_dict['set_title'], plot_dict['baseline']))