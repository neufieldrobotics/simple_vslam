#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
import numpy as np
np.set_printoptions(precision=4,suppress=True)


def zerrad(n,m,rho):
    R = 0.0
    for s in range(0,int((n-m)/2)+1):
         R = R + ((-1)**s*np.math.factorial(n-s)) / \
                 (np.math.factorial(s)* \
                  np.math.factorial(int((n+m)/2-s))* \
                  np.math.factorial(int((n-m)/2-s)) \
                 )* \
                 rho**(n-2*s)
    return R

def zerfilt(n,m,r):
    '''
    n and m integers that specify coeficient n > 0, m < n, m-n even
    r radius in pixels on image
    '''
    xdim = 2*r+1
    ydim = 2*r+1
    Z = np.zeros((xdim,ydim),dtype=complex)
    
    for y in range(-r,r+1):
      for x in range (-r,r+1):
        theta = np.arctan2(y,x)
        rho = ((x**2+y**2)**0.5)/r
        if rho <= 1:
          Z[y+r,x+r] = (n+1)/np.pi*np.exp(-m*theta*1j)*zerrad(n,m,rho)
    return Z

def zernike_generate(nmax,radius,verbose=True):
    '''
    generate the zernike filter to caluculate zernike coefficient n,m
    '''
    desc = 0
    Zfilt=[[None for x in range(nmax+1)] for y in range(nmax+1)]
    
    for n in range(nmax+1):
        for m in range(n%2,n+1,2):
            desc = desc+1
            if verbose: 
                print([radius, n, m, desc])
            Zfilt[n][m] = zerfilt(n,m,radius)
    return Zfilt,desc
     
'''
def plot_zernike(Z):
    #img = np.zeros((len(Z)*(Z[0][0]).shape[0],len(Z[0]*(Z[0][0]).shape[1])))
    img_list = []
    for x in Z:
        row = []
        for y in x:
            row.append()
'''
  