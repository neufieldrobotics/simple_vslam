#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:58:28 2019

@author: vik748
"""
import numpy as np
from matplotlib import pyplot as plt
import glob
import re
import os

def save_fig2png(fig, folder=None, fname=None):
    #plt._pylab_helpers.Gcf.figs.get(fig.number, None).window.showMaximized()
    plt.gcf().set_size_inches(np.array([16, 7.5])*1.5/2)
    plt.pause(.1)
    if fname is None:
        if fig._suptitle is None:
            fname = 'figure_{:d}'.format(fig.number)
        else:
            ttl = fig._suptitle.get_text()
            ttl = ttl.replace('$','').replace('\n','_').replace(' ','_')
            fname = re.sub(r"\_\_+", "_", ttl) 
    if folder:
        plt.savefig(os.path.join(folder, fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.png'))
    else:
        plt.savefig(file +'.png',format='png')

    plt._pylab_helpers.Gcf.figs.get(fig.number, None).window.showNormal()


files = sorted(glob.glob('/Users/vik748/Google Drive/data/feature_descriptor_comparision/results/raw_images_orbhc/*.csv'))
bins = np.linspace(10, 250, 25)

for file in files:
    results_array = np.genfromtxt(file, delimiter=',')
    lbl = ' '.join(os.path.splitext(os.path.basename(file))[0].split('_')[2:4])
    
    fig3 = plt.figure(3); plt.cla()
    
    plt.hist(results_array, bins=bins, alpha=0.5, label=['Zernike','ORB','ORB-Harris Corners'])
    plt.suptitle("Lars 1 800x600 Raw, ORB with Harris Corners - "+ lbl + " secs apart")
    plt.legend(loc='upper right')
    #plt.axes().set_ylim([0, 750])
    plt.xlabel('Bins (Number of matches)')
    plt.ylabel('Occurances (Image pairs)')
    plt.axes().set_ylim([0, 750])

    plt.show()
    
    save_fig2png(fig3)