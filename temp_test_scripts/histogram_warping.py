#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
histogra_warping

@author: vik748
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt, colors as clrs
from mpl_toolkits.mplot3d import Axes3D
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from vslam_helper import *
from matlab_imresize.imresize import imresize
from zernike.zernike import MultiHarrisZernike
import seaborn as sns ;# sns.set()
if os.path.dirname(os.path.realpath(__file__)) == os.getcwd():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data
import scipy.stats as st
import scipy.signal as sg
from scipy import interpolate as interp

def x_coord_lines(x_arr, y_arr,labels=None, ax = plt.gca(), *args, **kwargs ):
    for i,(x, y) in enumerate(zip(x_arr, y_arr)):
        ax.plot([x,x], [0,y],'o--',*args, **kwargs)
        if labels is not None:
            ax.annotate(labels+"{}".format(i), (x,0),*args, **kwargs)

def y_coord_lines(x_arr, y_arr,labels=None, ax = plt.gca(), *args, **kwargs ):
    for i,(x, y) in enumerate(zip(x_arr, y_arr)):
        ax.plot([0,x], [y,y],'o--',*args, **kwargs)
        if labels is not None:
            ax.annotate(labels+"{}".format(i), (0,y),*args, **kwargs)

def gen_F_inverse(F,x_d, delta = 1e-4):
    '''
    Given a cumulative F and gray values x_d
    '''
    zero_indices = np.where(F<delta)[0]
    last_zero = zero_indices[-1] if len(zero_indices)>0 else 0

    one_indices = np.where(F>(1-delta))[0]
    first_one = one_indices[0] if len(one_indices)>0 else len(F)

    F_trunc = np.copy(F[last_zero : first_one])
    F_trunc[0] = 0
    F_trunc[-1] = 1
    x_d_trunc = np.copy(x_d[last_zero : first_one])

    for f,x in zip(F_trunc, x_d_trunc):
        print(x,f)

    F_interp = interp.interp1d(F_trunc, x_d_trunc)
    return F_interp

#img1 = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261.png',1)
#img1_clahe = cv2.imread('/Users/vik748/Google Drive/data/contrast_test_images/G0286261_clahe.tif',1)

#gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#gr1_clahe = cv2.cvtColor(img1_clahe,cv2.COLOR_BGR2GRAY)

data_path = os.path.dirname(os.path.relpath(data.__file__))
img1 = cv2.imread(os.path.join(data_path,'histogram_warping_test_images','HistogramWarpBefore.png'),
                  cv2.IMREAD_COLOR)
img1_clahe = cv2.imread(os.path.join(data_path,'histogram_warping_test_images','HistogramWarpAfter.png'),
                        cv2.IMREAD_COLOR)


gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr1_clahe = cv2.cvtColor(img1_clahe,cv2.COLOR_BGR2GRAY)


plt.close('all')

fig1, fig1_axes = plt.subplots(3, 3, num=1, sharey='row')
fig1.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.1, hspace=0.11)
[axi.set_axis_off() for axi in fig1_axes[:2,:].ravel()]

fig1_axes[0,0].imshow(gr1,cmap='gray', vmin=0, vmax=255)
fig1_axes[0,0].set_title("RAW")

fig1_axes[1,0].imshow(gr1_clahe,cmap='gray', vmin=0, vmax=255)
fig1_axes[1,0].set_title("CLAHE")

h1,h1_bins,_ = fig1_axes[2,0].hist(gr1.flatten(), bins=np.linspace(0,255,256), density=True, alpha=0.4, label='Raw')

h1_clahe,_,_ = fig1_axes[2,0].hist(gr1_clahe.flatten(), bins=np.linspace(0,255,256), density=True, alpha=0.4, label='CLAHE')

[axi.legend() for axi in fig1_axes[2,:]]

fig2, fig2_axes = plt.subplots(3, 2, num=2)
fig2.clf()
fig2_axes[0,0] = fig2.add_subplot(3, 2, 1)
fig2_axes[1,0] = fig2.add_subplot(3, 2, 3)
fig2_axes[2,0] = fig2.add_subplot(3, 2, 5, sharex = fig2_axes[1,0])


fig2_axes[0,0].imshow(gr1,cmap='gray', vmin=0, vmax=255)
fig2_axes[0,0].set_title("RAW")
[axi.set_axis_off() for axi in fig2_axes[0,:].ravel()]



x_d = np.linspace(0,255,256)
x = gr1.flatten()

'''
from sklearn.neighbors import KernelDensity
h = np.std(x)*(4/3/len(x))**(1/5)
h = 0.7816774 * st.iqr(x) * ( len(x) ** (-1/7))
plt.hist(gr1.flatten(), bins=np.linspace(0,255,256), density=True, alpha=0.4, label='Raw')

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=h, kernel='gaussian')
kde.fit(x[:,None])
logprob = kde.score_samples(x_d[:, None])
'''

fig2_axes[1,0].hist(gr1.flatten(), bins=np.linspace(0,255,256), color='blue', density=True, alpha=0.4, label='Raw')

x_kde_full = st.gaussian_kde(x,bw_method='silverman')
x_kde = x_kde_full(x_d)

fig2_axes[1,0].fill_between(x_d, x_kde, color='red',alpha=0.4)
#plt.ylim(-0.02, 0.22)
valleys, = sg.argrelmin(x_kde)
v_k = np.concatenate( (np.array([0]), valleys,np.array([255]) ) )
a_k = np.round((v_k[0:-1] + v_k[1:])/2).astype(int)


fig2_axes[1,0].plot(a_k, x_kde[a_k],'o',label='a_k')
fig2_axes[1,0].set_xlim(0,255)



f = x_kde
F = np.cumsum(f)

x_coord_lines(v_k, f[v_k],labels='v_', ax=fig2_axes[1,0], color='y')
x_coord_lines(a_k, f[a_k],labels='a_', ax=fig2_axes[1,0], color='g')


fig2_axes[2,0].hist(gr1.flatten(), bins=np.linspace(0,255,256), color='blue',
                  cumulative=True, density=True, alpha=0.4, label='Raw')
fig2_axes[2,0].fill_between(x_d,F,color='red',alpha=0.4)
x_coord_lines(v_k, F[v_k],labels='v_', ax=fig2_axes[2,0], color='y')
y_coord_lines(v_k, F[v_k],labels='Fv_', ax=fig2_axes[2,0], color='y')

x_coord_lines(a_k, F[a_k],labels='a_', ax=fig2_axes[2,0], color='g')
y_coord_lines(a_k, F[a_k],labels='Fa_', ax=fig2_axes[2,0], color='g')


vk = v_k[1:]
vk1 = v_k[0:-1]
b_k = ( (F[vk] - F[a_k])* vk1 + (F[a_k]-F[vk1]) * vk ) / (F[vk] - F[vk1])

x_coord_lines(b_k, f[a_k],labels='b_', ax=fig2_axes[1,0], color='b')

'''
vk = v_k[1]
vk1 = v_k[0]
ak = a_k[0]
'''

F_interp = gen_F_inverse(F,x_d, delta = 1e-4)

a_k_full = np.concatenate( (np.round(F_interp([0])).astype(int), a_k, np.round(F_interp([1])).astype(int) ) )
a_k_plus_1 = a_k_full[2:]
a_k_minus_1 = a_k_full[0:-2]


a_k_plus = F_interp((F[a_k] + F[a_k_plus_1])/2)    #.astype(int)
a_k_minus = F_interp((F[a_k] + F[a_k_minus_1])/2)  #.astype(int)

b_k_full = np.concatenate( (np.array([0]), b_k, np.array([255]) ) )
b_k_plus_1 = b_k_full[2:]
b_k_minus_1 = b_k_full[0:-2]

b_k_plus = ( b_k + b_k_plus_1 ) / 2
b_k_minus = ( b_k + b_k_minus_1 ) / 2