#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:26:56 2019

@author: vik748
"""

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057821.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
img2 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057826.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

gr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

Nfeats  = 600        # number of features per image
seci    = 2          # number of vertical sectors 
secj    = 3          # number of horizontal sectors
levels  = 6        # pyramid levels
ratio   = 0.75        # scaling between levels
sigi    = 2.75          # integration scale 1.4.^[0:7];%1.2.^[0:10]
sigd    = 1.0          # derivation scale
lmax_nd = 3       # Feature neighborhood size for local maximum filter

def xy_gradients(img):
    kernelx = 1/2*np.array([[-1,0,1]])
    kernely = 1/2*np.array([[-1],[0],[1]])
    fx = cv2.filter2D(lpf,cv2.CV_32F,kernelx)
    fy = cv2.filter2D(lpf,cv2.CV_32F,kernely)
    return fx, fy

'''
Generate image pyramid, based on settings in the MultiHarrisZernike object
'''
sigd_list = [sigd]
sigi_list = [sigi]
images = [np.float32(gr1)]

# convolve matches matlab version better, filter is 3 times faster
# Reduce Ksize to improve speed
lpimages = [cv2.GaussianBlur(images[0], ksize=(7,7), sigmaX=1.0, sigmaY=1.0, borderType = cv2.BORDER_CONSTANT)]

for k in range(1,levels):
    # CV2 version of imresize is faster but doesn't have antialiasing
    # so results in fewer matches
    images += [cv2.resize(images[-1], (0,0), fx=ratio,
                          fy=ratio, interpolation=cv2.INTER_AREA)]
    #images += [cv2.resize(images[-1], (np.ceil(images[-1].shape[1]*ratio).astype(int),np.ceil(images[-1].shape[0]*ratio).astype(int)),
    #                      interpolation=cv2.INTER_AREA)]
    
    # convolve matches matlb version better, filter is 3 times faster
    lpimages += [cv2.GaussianBlur(images[-1], ksize=(7,7), sigmaX=1.0, sigmaY=1.0, borderType = cv2.BORDER_CONSTANT)]
    

    sigd_list += [sigd_list[-1]/ratio] #equivalent sigdec at max res
    sigi_list += [sigi_list[-1]/ratio] 

scale = 1

[fy,fx] = np.gradient(lpf)

[fxy,fxx] = np.gradient(fx)
[fyy,fyx] = np.gradient(fy)
nL = scale**(-2)*np.abs(fxx+fyy)

Mfxx = cv2.GaussianBlur(np.square(fx), ksize=(11,11), sigmaX=sigi, sigmaY=sigi, borderType = cv2.BORDER_CONSTANT)
Mfxy = cv2.GaussianBlur(fx * fy, ksize=(11,11), sigmaX=sigi, sigmaY=sigi, borderType = cv2.BORDER_CONSTANT)
Mfyy = cv2.GaussianBlur(np.square(fy), ksize=(11,11), sigmaX=sigi, sigmaY=sigi, borderType = cv2.BORDER_CONSTANT)

Tr = Mfxx + Mfyy
Det = Mfxx * Mfyy -np.square(Mfxy)
sqrterm = np.sqrt(np.square(Tr) - 4*Det)

ef2 = scale**(-2)*0.5*(Tr - sqrterm)
