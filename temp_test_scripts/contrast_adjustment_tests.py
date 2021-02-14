#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_Pose_test

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


def adj_contrast(gr, alpha, beta=0):
    '''
    Adjust contrast of the image using slope alpha and brightness beta
    where f(x)=alpha((x−128) + 128 + beta
    '''
    assert gr.ndim == 2, "Number of image dims != 2, possibly rgb image"
    return np.round(alpha*(gr-128.0)+128+beta).astype(np.uint8)

def hist_eq(gr):
    '''
    Return histogram equalized image
    '''



'''
resize_ratio = 1/5

img1 = cv2.imread('/Users/vik748/Google Drive/data/chess_board/GOPR1488.JPG',1)          # queryImage
img2 = cv2.imread('/Users/vik748/Google Drive/data/chess_board/GOPR1490.JPG',1)

fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);

if resize_ratio != 1:
    img1 = imresize(img1, resize_ratio, method='bicubic')
    img2 = imresize(img2, resize_ratio, method='bicubic')
    K = K * resize_ratio

tiling = {'x': 4, 'y': 3, 'no_features': NO_OF_FEATURES }



### KITTI DATA
img1 = cv2.imread('/Users/vik748/Google Drive/data/kitti/00/image_0/000001.png',1)          # queryImage
img2 = cv2.imread('/Users/vik748/Google Drive/data/kitti/00/image_0/000003.png',1)

K = np.float64([ [718.856,   0.0  , 607.1928],
                 [  0.0  , 718.856, 185.2157],
                 [  0.0  ,   0.0  ,   1.0   ] ])
D = np.float64([0.0, 0.0, 0.0, 0.0, 0.0])
tiling = None #{'x': 17, 'y': 8, 'no_features': NO_OF_FEATURES }


#img1 = cv2.imread(path+'/data/Cervino_1_080618_800x600/G0011701.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img2 = cv2.imread(path+'/data/Cervino_1_080618_800x600/G0011741.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

img1 = cv2.imread('/Users/vik748/Google Drive/data/Cervino_1_080618_800x600/G0010601.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
img2 = cv2.imread('/Users/vik748/Google Drive/data/Cervino_1_080618_800x600/G0010621.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE


K = np.float64([[ 704.7828555610147,   0.0            , 401.4192884115758  ],
                [   0.0            , 697.4874654750291, 296.63918937437165 ],
                [   0.0            ,   0.0            ,   1.0              ]])

D = np.float64([-0.28574540724519565, 0.15949992494106607, -0.000515563796390175, -9.00485425041488e-05, -0.0743708876047786])
'''

img1 = cv2.imread('/Users/vik748/Google Drive/data/Lars1_080818_800x600/G0285493.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
img2 = cv2.imread('/Users/vik748/Google Drive/data/Lars1_080818_800x600/G0285513.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

K = np.float64([[ 704.1386022291683,   0.0           , 405.8879561895266 ],
                [   0.0            , 686.904621157287, 296.6670180431998 ],
                [   0.0            ,   0.0           ,   1.0             ]])

D = np.float64([-0.279351255654882336, 0.127088911708472863, -0.000495524072154332213, 0.0000424940878375506994, 0.0278131704111216453])




NO_OF_FEATURES = 1200
tiling=None
print(K,D)





# Convert images to greyscale
gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)



#detector = cv2.ORB_create(nfeatures=2 * NO_OF_FEATURES, edgeThreshold=31, patchSize=31, nlevels=8,
#                          fastThreshold=15, scaleFactor=1.2, WTA_K=2,
#                          scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

detector = MultiHarrisZernike(Nfeats= 1200, seci= 4, secj= 3, levels= 6, ratio= 1/1.2,
                              sigi= 2.75, sigd= 1.0, nmax= 8, like_matlab= False, lmax_nd= 3, harris_threshold = 0.01)


descriptor = cv2.ORB_create(nfeatures=2 * NO_OF_FEATURES, edgeThreshold=31, patchSize=31, nlevels=6,
                             fastThreshold=15, scaleFactor=1.2, WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

fig1, fig1_axes = plt.subplots(2, 2, num=1, sharey='row')
fig2, fig2_axes = plt.subplots(2, 2, num=2, sharey='row')

fig1.suptitle('800x600 Raw Images Top 25 features')
h1_vals,_,_ = fig1_axes[1,0].hist(gr1.flatten(), bins=np.linspace(0,255,52), alpha=0.5)
fig1.subplots_adjust(left=0.1, bottom=0.1, right=1.0, top=.9, wspace=0.1, hspace=0.0)

match_image_pairs(detector, detector, (gr1,gr2), K=K, D=D, tiling=tiling, pixel_matching_dist = None,
                  lowe_threshold = 0.9,
                  feat_img_axes = fig1_axes[0,0], matching_img_axes = fig2_axes[0,0],
                  match_dist_plot_axes = fig2_axes[1,0])

fig2.subplots_adjust(left=0.05, bottom=0.1, right=1.0, top=.9, wspace=0.1, hspace=0.0)


gr1_adj = adj_contrast(gr1, alpha=0.125, beta=0)
gr2_adj = adj_contrast(gr2, alpha=0.125, beta=0)


match_image_pairs(detector, detector, (gr1_adj,gr2_adj), K=K, D=D, tiling=tiling, pixel_matching_dist = 100,
                  feat_img_axes = fig1_axes[0,1], matching_img_axes = fig2_axes[0,1],
                  match_dist_plot_axes = fig2_axes[1,1])

fig1_axes[0,1].axis("off")
fig1_axes[0,1].imshow(gr1_adj,cmap='gray', vmin=0, vmax=255)
fig2_axes[0,1].axis("off")

h2_vals,_,_ = fig1_axes[1,1].hist(gr1_adj.flatten(), bins=np.linspace(0,255,52), alpha=0.5)

#ylim_max = np.round(np.max(np.concatenate((h1_vals,h2_vals)))*1.1,decimals=-3)
#fig1_axes[1,0].set_ylim([0, ylim_max])
fig1_axes[1,0].set_yticklabels(['{:,}'.format(x / 1000) for x in fig1_axes[1,0].get_yticks()])
#fig1_axes[1,1].set_ylim([0, ylim_max])


hist,bin = np.histogram(gr1_adj.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * 255 / cdf.max()

plt.figure(4)
plt.plot(cdf_normalized, color = 'b')
#plt.hist(gr1_adj.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf_new = np.ma.filled(cdf_m,0).astype('uint8')
plt.plot(cdf_new, color = 'g')


