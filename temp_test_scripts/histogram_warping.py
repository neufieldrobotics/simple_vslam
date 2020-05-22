#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
histogra_warping

@author: vik748
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
if os.path.dirname(os.path.realpath(__file__)) == os.getcwd():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data
import scipy.stats as st
from scipy.signal import argrelmin
from scipy import interpolate as interp
from scipy import integrate

def x_coord_lines(x_arr, y_arr,labels=None, ax = plt.gca(), *args, **kwargs ):
    for i,(x, y) in enumerate(zip(x_arr, y_arr)):
        ax.plot([x,x], [0,y],'o--',*args, **kwargs)
        if labels is not None:
            ax.annotate(labels+"{}".format(i), (x,y/2),*args, **kwargs)

def y_coord_lines(x_arr, y_arr,labels=None, ax = plt.gca(), *args, **kwargs ):
    for i,(x, y) in enumerate(zip(x_arr, y_arr)):
        ax.plot([0,x], [y,y],'o--',*args, **kwargs)
        if labels is not None:
            ax.annotate(labels+"{}".format(i), (x/2,y),*args, **kwargs)

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

    #for f,x in zip(F_trunc, x_d_trunc):
    #    print(x,f)

    F_interp = interp.interp1d(F_trunc, x_d_trunc)
    return F_interp

def get_Transform(a,b,d,x):
    '''
    a = array of segment mid points
    b = location of the adjusted mid points
    d = slops
    '''

    assert len(a) == len(b) == len(d)

    x_map =  np.array([])
    T_x =  np.array([])
    for k in range(1,len(a)):
        r_k = ( b[k] - b[k-1] ) / ( a[k] - a[k-1] )
        x_in =  x[np.logical_and(x > a[k-1], x < a[k])]
        t = ( x_in - a[k-1] ) /  (a[k] - a[k-1] )
        T = b[k-1] + \
            ( ( r_k * t**2 + d[k-1]*(1-t)*t ) * ( b[k] - b[k-1] ) /
              ( r_k + ( d[k] + d[k-1] - 2*r_k ) * (1-t) * t ) )

        x_map = np.concatenate((x_map, x_in))
        T_x = np.concatenate((T_x, T))

    return interp.interp1d(x_map, T_x,
                           bounds_error = False,
                           fill_value = ( np.min(T_x), np.max(T_x) ) )

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
fig2 = plt.figure(2, constrained_layout=True)
gs = fig2.add_gridspec(4, 2)
fig2_axes[0,0] = fig2.add_subplot(gs[0:2,0])
fig2_axes[1,0] = fig2.add_subplot(gs[2,0])
fig2_axes[2,0] = fig2.add_subplot(gs[3,0], sharex = fig2_axes[1,0])

fig2_axes[0,1] = fig2.add_subplot(gs[0:2,1])
fig2_axes[1,1] = fig2.add_subplot(gs[2,1], sharey = fig2_axes[1,0])
fig2_axes[2,1] = fig2.add_subplot(gs[3,1], sharex = fig2_axes[1,1], sharey = fig2_axes[2,0])

fig2_axes[0,0].imshow(gr1,cmap='gray', vmin=0, vmax=255)
fig2_axes[0,0].set_title("RAW")
[axi.set_axis_off() for axi in fig2_axes[0,:].ravel()]

no_bits = 8
no_gray_levels = 2 ** no_bits

x = np.linspace(0,1, no_gray_levels)
x_img = gr1.flatten() / (no_gray_levels - 1)

fig2_axes[1,0].hist(x_img, bins=x, color='blue', density=True, alpha=0.4, label='Raw')

#h = 0.7816774 * st.iqr(x_img) * ( len(x_img) ** (-1/7) )
#h = 0.7816774 * ( Finv_interp(0.75) - Finv_interp(0.25) ) * ( len(x_img) ** (-1/7) )
x_kde_full = st.gaussian_kde(x_img,bw_method='silverman')
x_kde = x_kde_full(x)

f = x_kde
F = np.cumsum(f)
F = np.concatenate((np.array([0]), integrate.cumtrapz(f, x)))

f_interp = interp.interp1d(x, f)
F_interp = interp.interp1d(x, F)
Finv_interp = gen_F_inverse(F,x, delta = 1e-4)

fig2_axes[1,0].fill_between(x, f, color='red',alpha=0.4)
fig2_axes[1,0].set_xlim(0,1)

valleys = x[argrelmin(f)[0]]
v_k = np.concatenate( (np.array([0]), valleys,np.array([1]) ) )
a_k = (v_k[0:-1] + v_k[1:])/2


fig2_axes[1,0].plot(a_k, f_interp(a_k),'o',label='a_k')


x_coord_lines(v_k, f_interp(v_k),labels='v_', ax=fig2_axes[1,0], color='y')
x_coord_lines(a_k, f_interp(a_k),labels='a_', ax=fig2_axes[1,0], color='g')


fig2_axes[2,0].hist(x_img, bins=x, color='blue', cumulative=True,
                    density=True, alpha=0.4, label='Raw')
fig2_axes[2,0].fill_between(x, F, color='red',alpha=0.4)

x_coord_lines(v_k, F_interp(v_k),labels='v_', ax=fig2_axes[2,0], color='y')
y_coord_lines(v_k, F_interp(v_k),labels='Fv_', ax=fig2_axes[2,0], color='y')

x_coord_lines(a_k, F_interp(a_k),labels='a_', ax=fig2_axes[2,0], color='g')
y_coord_lines(a_k, F_interp(a_k),labels='Fa_', ax=fig2_axes[2,0], color='g')


vk = v_k[1:]
vk1 = v_k[0:-1]
b_k = ( (F_interp(vk)  - F_interp(a_k)) * vk1 +
        (F_interp(a_k) - F_interp(vk1)) * vk  ) / \
      (  F_interp(vk)  - F_interp(vk1) )

x_coord_lines(b_k, f_interp(a_k),labels='b_', ax=fig2_axes[1,0], color='b')
x_coord_lines(b_k, F_interp(a_k),labels='b_', ax=fig2_axes[2,0], color='b')


# Calculate dk
a_k_full = np.concatenate( ( Finv_interp([0]), a_k, Finv_interp([1]) ) )
a_k_plus_1 = a_k_full[2:]
a_k_minus_1 = a_k_full[0:-2]

a_k_plus = Finv_interp((F_interp(a_k) + F_interp(a_k_plus_1))/2)    #.astype(int)
a_k_minus = Finv_interp((F_interp(a_k) + F_interp(a_k_minus_1))/2)  #.astype(int)

b_k_full = np.concatenate( (np.array([0]), b_k, np.array([1]) ) )
b_k_plus_1 = b_k_full[2:]
b_k_minus_1 = b_k_full[0:-2]

b_k_plus = ( b_k + b_k_plus_1 ) / 2
b_k_minus = ( b_k + b_k_minus_1 ) / 2

exp_denom = F_interp(a_k_plus_1) - F_interp(a_k_minus_1)

first_term = ( (b_k - b_k_minus) / (a_k - a_k_minus) ) ** \
             ( ( F_interp(a_k) - F_interp(a_k_minus_1) ) / exp_denom )

second_term = ( (b_k_plus - b_k) / (a_k_plus - a_k) ) ** \
              ( ( F_interp(a_k_plus_1) - F_interp(a_k) ) / exp_denom )

d_k = first_term * second_term

b_0_plus = ( b_k_full[0] + b_k_full[1] ) / 2
a_0_plus = Finv_interp( ( F_interp(a_k_full[0]) + F_interp(a_k_full[1]) )/2 )
d_0 = b_0_plus / a_0_plus

b_K_plus_1_minus = ( b_k_full[-1] - b_k_full[-2] ) / 2
a_K_plus_1_minus = Finv_interp( ( F_interp(a_k_full[-1]) - F_interp(a_k_full[-2]) )/2 )
d_K_plus_1 = ( 1 - b_K_plus_1_minus ) / ( 1 - a_K_plus_1_minus )

d_k_full = np.concatenate( (np.array([d_0]), b_k, np.array([d_K_plus_1]) ) )

lam = 5.0
d_k_full[d_k_full < 1/lam] = lam
d_k_full[d_k_full > lam]

T_x_interp = get_Transform(a_k_full, b_k_full, d_k_full, x)

x_img_adj = T_x_interp(x_img)
gr1_warp = np.round(x_img_adj * (no_gray_levels - 1)).astype(int).reshape(gr1.shape)

fig2_axes[0,1].imshow(gr1_warp,cmap='gray', vmin=0, vmax=255)
fig2_axes[0,1].set_title("Histogram Warped")

fig2_axes[1,1].hist(x_img_adj, bins=x, color='blue', density=True, alpha=0.4, label='Warped')
fig2_axes[1,1].set_xlim(0,1)

x_adj_kde_full = st.gaussian_kde(x_img_adj, bw_method=x_kde_full.silverman_factor())
x__adj_kde = x_adj_kde_full(x)

f_adj = x__adj_kde
F_adj = np.cumsum(f_adj)
F_adj = np.concatenate((np.array([0]), integrate.cumtrapz(f_adj, x)))

fig2_axes[1,1].fill_between(x, f_adj, color='red',alpha=0.4)


fig2_axes[2,1].hist(x_img_adj, bins=x, color='blue', cumulative=True,
                    density=True, alpha=0.4, label='Raw')

fig2_axes[2,1].fill_between(x, F_adj, color='red',alpha=0.4)

x_coord_lines(v_k, f_interp(v_k),labels='v_', ax=fig2_axes[1,1], color='y')
x_coord_lines(a_k, f_interp(a_k),labels='a_', ax=fig2_axes[1,1], color='g')

x_coord_lines(v_k, F_interp(v_k),labels='v_', ax=fig2_axes[2,1], color='y')
y_coord_lines(v_k, F_interp(v_k),labels='Fv_', ax=fig2_axes[2,1], color='y')

x_coord_lines(a_k, F_interp(a_k),labels='a_', ax=fig2_axes[2,1], color='g')
y_coord_lines(a_k, F_interp(a_k),labels='Fa_', ax=fig2_axes[2,1], color='g')
x_coord_lines(b_k, f_interp(a_k),labels='b_', ax=fig2_axes[1,1], color='b')
x_coord_lines(b_k, F_interp(a_k),labels='b_', ax=fig2_axes[2,1], color='b')

