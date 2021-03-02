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

def dfplot_with_trendline(df, x_col_name, y_col_name, ax=None, color=None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    df.plot.scatter(x=x_col_name, y=y_col_name, marker='.', s=16, c=color, ax=ax, label=x_col_name)
    line_coeff = np.polyfit(df[x_col_name], df[y_col_name], 1)
    line_polt1d = np.poly1d(line_coeff)
    ax.plot(df[x_col_name], line_polt1d(df[x_col_name]),'--',color=color,label=x_col_name+" trend")

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/data'
else:
    path = os.path.expanduser('~/data/')

results_files = [#'results/matching_results_20200710_092947_Lars1_full.csv',
                 #'results/matching_results_20200711_053758_Lars2_full.csv',
                 '../results/matching_results_20200720_143024_Stingray2_with_mask_names_fixed.csv',
                 '../results/matching_results_20200722_003934-Morgan1_072719_800x600.csv',
                 '../results/matching_results_20200721_025524-Morgan2_073019_800x600.csv']

#results_file = 'results/matching_results_20200721_025524 - Morgan2_073019_800x600.csv'
#results_file = max(glob.glob('results/matching*.csv'), key=os.path.getctime)

#results_df = pd.read_csv(results_file)
results_dfs = [pd.read_csv(rf) for rf in results_files]
#results_df2 = pd.read_csv(results_file2)
#results_df3 = pd.read_csv(results_file3)

results_df = pd.concat(results_dfs, ignore_index=True)

#results_df.loc[(results_df['detector'] == 'ORB') && (results_df['descriptor'] == 'ORB')]
contrast_adj_factors = np.array([0.0]) # np.arange(0,-1.1,-.2)


detdes_dict_list = [{'detector': 'ORB', 'descriptor': 'ORB'},
                    {'detector': 'MultiHarrisZernike', 'descriptor': 'ORB'},
                    {'detector': 'MultiHarrisZernike', 'descriptor': 'MultiHarrisZernike'}]

ctrst_meas_names = ['rms_contrast','local_box_filt','local_bilateral_filt','local_gaussian_filt','global_contrast_factor' ]

for ctrst_meas_name in ctrst_meas_names:
    fig,axes = plt.subplots(1,3, sharex=True, sharey=True)

    for detdes_dict, ax in zip(detdes_dict_list, axes):

        for cont_fact in contrast_adj_factors:
            plot_dict = { **detdes_dict,
                         #'set_title':'Morgan1_072719_800x600',
                         'contrast_adj_factor': cont_fact,
                         'baseline': 20}

            bool_series = pd.Series(index=results_df.index, dtype=bool,data=1)

            for it, val in plot_dict.items():
                bool_series = bool_series & ( results_df[it] == val )

            #relevant_matches = results_df[bool_series]['matches'].values

            relevant_df = results_df[bool_series]
            relevant_df['mean_'+ctrst_meas_name]=relevant_df[['img0_'+ctrst_meas_name, 'img1_'+ctrst_meas_name]].mean(axis=1)
            relevant_df['mean_'+ctrst_meas_name+'_masked']=relevant_df[['img0_'+ctrst_meas_name+'_masked', 'img1_'+ctrst_meas_name+'_masked']].mean(axis=1)

            dfplot_with_trendline(relevant_df,'mean_'+ctrst_meas_name, 'matches',ax=ax)
            if not relevant_df['mean_'+ctrst_meas_name+'_masked'].isnull().all():
                dfplot_with_trendline(relevant_df,'mean_'+ctrst_meas_name+'_masked', 'matches',ax=ax,color='tab:orange')

        ax.set_title("Det: {} Des: {}".format(plot_dict['detector'],plot_dict['descriptor']))
        ax.set_xlabel('Contrast Measurement')
        ax.legend()

    axes[0].set_ylabel('Number of matches')
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.05, hspace=0.05)
    fig.suptitle("Dataset: {}, baseline: {} Contrast Meas: {}".format(None, plot_dict['baseline'],ctrst_meas_name))
    save_fig2png(fig, size=[16.0, 4.875])