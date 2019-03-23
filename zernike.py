#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
import numpy as np
from matlab_imresize.imresize import imresize
np.set_printoptions(precision=5,suppress=True)
from scipy.ndimage import convolve
from scipy.ndimage.filters import maximum_filter
import sys
import cv2
from matplotlib import pyplot as plt

class MultiHarrisZernike (cv2.Feature2D):
    '''
    MultiHarrisZernike feature detector which uses multi-level harris corners
    along with Zernike parameters on 2 different radii discs as the feature detector
    A class as a child of cv2.Feature2D
    
    Example usage:
        img = cv2.imread('test.png',1)
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        a = MultiHarrisZernike()
        kp, des = a.detectAndCompute(gr)
        
        outImage	 = cv2.drawKeypoints(gr, kp, gr,color=[255,255,0],
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        fig, ax= plt.subplots(dpi=200)
        plt.title('Multiscale Harris with Zernike Angles')
        plt.axis("off")
        plt.imshow(outImage)
        plt.show()
    
    '''
    def __init__(self,  Nfeats= 600, seci = 2, secj = 3, levels = 6,
                 ratio = 0.75, sigi = 2.75, sigd = 1, nmax = 8, maxdes = (12.0,8.0)):       
        self.Nfeats = Nfeats    # number of features per image
        self.seci   = seci      # number of vertical sectors 
        self.secj   = secj      # number of horizontal sectors
        self.levels = levels    # pyramid levels
        self.ratio  = ratio     # scaling between levels
        self.sigi   = sigi      # integration scale 1.4.^[0:7];%1.2.^[0:10]
        self.sigd   = sigd      # derivation scale
        self.nmax   = nmax      # zernike order
        self.maxdes = maxdes    # The factor used to convert the Float descriptors to UINT8
        self.zrad   = np.ceil(self.sigi*8).astype(int) # radius for zernike disk  
        self.brad   = np.ceil(0.5*self.zrad).astype(int)    # radius for secondary zernike disk
        self.Gi     = MultiHarrisZernike.fspecial_gauss(11,self.sigi)
        self.pyrlpf = MultiHarrisZernike.fspecial_gauss(int(np.ceil(7*self.sigd)),self.sigd)
        self.ZstrucZ, self.ZstrucNdesc = MultiHarrisZernike.zernike_generate(self.nmax, self.zrad)
        self.BstrucZ, self.BstrucNdesc = MultiHarrisZernike.zernike_generate(self.nmax, self.brad)
        
    @staticmethod
    def fspecial_gauss(size, sigma):
        """
        Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()
    
    @staticmethod
    def zernike_generate(nmax,radius,verbose=False):
        '''
        generate the zernike filter to caluculate zernike coefficient n,m
        '''
        desc = 0
        Zfilt=[[0 for x in range(nmax+1)] for y in range(nmax+1)]
        
        for n in range(nmax+1):
            for m in range(n%2,n+1,2):
                desc = desc+1
                if verbose: 
                    print([radius, n, m, desc])
                Zfilt[n][m] = MultiHarrisZernike.zerfilt(n,m,radius)
        return Zfilt,desc             
    
    @staticmethod
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
                    Z[y+r,x+r] = (n+1)/np.pi*np.exp(-m*theta*1j)*MultiHarrisZernike.zerrad(n,m,rho)
        return Z
    
    @staticmethod
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

    @staticmethod    
    def plot_zernike(Z):
        '''
        Plot the generated zernike polynomials
        '''
        nm=len(Z)
        f, axes = plt.subplots(nm, nm, sharey=True)
        #f.subplots_adjust(0,0,1,1)
        f.subplots_adjust(wspace=0, hspace=0)
    
        w, h = Z[0][0].shape
        for n in range(nm):
            for m in range(nm):
                if Z[n][m] is not 0:
                    axes[n,m].imshow(np.real(Z[n][m]),cmap='gray')
                axes[n,m].axis('off')
                
    def generate_pyramid(self,img):
        '''
        Generate image pyramid, based on settings in the MultiHarrisZernike object
        '''
        sigd_list = [self.sigd]
        sigi_list = [self.sigi]
        images = [np.float64(img)]
        lpimages = [convolve(images[0],self.pyrlpf,mode='constant')]
        for k in range(1,self.levels):
            #self.images += [cv2.resize(self.images[-1], (0,0), fx=self.ratio,
            #                          fy=self.ratio, interpolation=cv2.INTER_LINEAR_EXACT)]
            images += [imresize(images[-1], self.ratio, method='bilinear')]
            lpimages += [convolve(images[-1],self.pyrlpf,mode='constant')]
            sigd_list += [sigd_list[-1]/self.ratio] #equivalent sigdec at max res
            sigi_list += [sigi_list[-1]/self.ratio] 
        return {'images':images, 'lpimages':lpimages, 'sigd':sigd_list, 'sigi':sigi_list}
    
    def eigen_image_p(self,lpf,scale):
        '''    
        ef2,nL = eigen_image_p(lpf,scale)
        set up in pyramid scheme with detection scaled smoothed images
        ef2 is the interest point eigen image
        lpf smoothed by the detection scale gaussian
        Gi = fspecial('gaussian',ceil(7*sigi),sigi);
        '''
        [fy,fx] = np.gradient(lpf)
    
        [fxy,fxx] = np.gradient(fx)
        [fyy,fyx] = np.gradient(fy)
        nL = scale**(-2)*np.abs(fxx+fyy)
    
        Mfxx = convolve(np.square(fx),self.Gi,mode='constant')
        Mfxy = convolve(fx*fy,self.Gi,mode='constant')
        Mfyy = convolve(np.square(fy),self.Gi,mode='constant')
    
        Tr = Mfxx+Mfyy
        Det = Mfxx*Mfyy-np.square(Mfxy)
        sqrterm = np.sqrt(np.square(Tr)-4*Det)
    
        ef2 = scale**(-2)*0.5*(Tr - sqrterm)
        return ef2,nL     

    def feat_extract_p2 (self,ImgPyramid):
        '''
        Extract multiscaled features from a Pyramid of images
        '''
        local_maxima_nhood=3
    
        scales = self.levels
        ratio = self.ratio
        border = self.zrad
        lpimages = ImgPyramid['lpimages']
        [rows,cols] = lpimages[0].shape
    
        eig = [None] * scales
        nL = [None] * scales
        border_mask = [None] * scales 
        regmask=[None] * scales 
        ivec= [None] * scales 
        jvec= [None] * scales
    
        for k in range(scales):
            [eig[k], nL[k]] = self.eigen_image_p(lpimages[k],ratio**(k))
            # extract regional max and block out borders (edge effect)
    
            # generate mask for border
            border_mask[k] = np.zeros_like(eig[k],dtype=bool)
            border_mask[k][border:-border,border:-border]=True
    
            regmask[k] = maximum_filter(eig[k],size=local_maxima_nhood)<=eig[k]
    
            regmask[k] = np.logical_and(regmask[k],border_mask[k])
            #print("K: ",k," - ",np.sum(regmask[k]))
            #[ivec[k], jvec[k]] = np.nonzero(regmask[k]) #coordinates of 1s in regmask
            # Just to match matlab version, can be reverted to optimise
            [jvec[k], ivec[k]] = np.nonzero(regmask[k].T)
    
        # INITIALIZE feature positions and scales at highest level
        # at highest resolution coordinates of features:
        Fivec = ivec[0]
        #print("len of Fivec:",len(ivec[0]))
        Fjvec = jvec[0]
        Fsvec = np.zeros_like(Fivec) #initial scale 
        Fevec = eig[0][ivec[0],jvec[0]] #access the elements of eig at locations given by ivec,jvec
    
        #i,j position of feature at the characteristic scale
        Fsivec = np.copy(Fivec)
        Fsjvec = np.copy(Fjvec)
    
        nLvec = nL[0][ivec[0],jvec[0]]
        pivec = np.copy(Fivec)
        pjvec = np.copy(Fjvec)
        pind = np.array(list(range(len(Fivec))))
        k = 1
    
        while  (k < scales) & (len(pivec) > 0):
            mx = (np.floor(cols*ratio)-1)/(cols-1)  #scale conversion to next level
            my = (np.floor(rows*ratio)-1)/(rows-1) 
    
            [rows,cols]  = eig[k].shape #dimensions of next level
            pendreg = np.zeros_like(pivec)
            # match matlab output
            sivec = np.round(pivec*my+np.finfo(np.float32).eps).astype(int) #next scale ivec
            sjvec = np.round(pjvec*mx+np.finfo(np.float32).eps).astype(int) #next scale jvec
    
            csivec = np.copy(sivec)
            csjvec = np.copy(sjvec)
      
            for u in range(-1,2):  #account for motion of feature points between scales
                for v in range(-1,2):
                    sojvec = sjvec+u #next scale jvec
                    soivec = sivec+v #next scale ivec
                    uvpend = regmask[k][soivec,sojvec] == 1
                    pendreg = np.logical_or(pendreg,uvpend)
                    csivec[uvpend] = soivec[uvpend]
                    csjvec[uvpend] = sojvec[uvpend]
    
            pend = np.logical_and(pendreg, nL[k][csivec,csjvec] >= nLvec)
            pind = pind[pend]
    
            Fsvec[pind] = k #scale is k or larger
            Fevec[pind] = eig[k][csivec[pend],csjvec[pend]] #eigen value is given at
                                                 #level k or larger
            Fsivec[pind] = csivec[pend]
            Fsjvec[pind] = csjvec[pend]
    
            pivec = csivec[pend]
            pjvec = csjvec[pend]
            nLvec = nL[k][csivec[pend],csjvec[pend]]
            #print(np.sum(Fsvec==k))
            k = k+1
            F = {'ivec':Fivec, 'jvec':Fjvec, 'svec':Fsvec,
                 'evec':Fevec, 'sivec':Fsivec, 'sjvec':Fsjvec}
        return F

    def feat_thresh_sec(self,F,rows,cols):
     
        Nsec = self.seci*self.secj
        Nfsec = np.ceil(self.Nfeats/Nsec).astype(int)
        
        seclimi = np.linspace(0,rows-1,self.seci+1)
        seclimj = np.linspace(0,cols-1,self.secj+1)
        
        Fivec = F['ivec']
        Fjvec = F['jvec']
        Fevec = F['evec']
        Fsvec = F['svec']
        Fsivec = F['sivec']
        Fsjvec = F['sjvec']
        select = np.array([],dtype=int) #zeros(size(F.ivec))
        selind = np.array(list(range(len(Fivec))))
        for i_ll,i_ul in zip(seclimi[:-1],seclimi[1:]):
            selecti = np.logical_and( Fivec >= i_ll , Fivec < i_ul)
            for j_ll,j_ul in zip(seclimj[:-1],seclimj[1:]):
                selectj = np.logical_and(Fjvec >= j_ll, Fjvec < j_ul)
                selectsec = np.logical_and(selecti, selectj)
                evec = Fevec[selectsec]
                selindsec = selind[selectsec]
                N,bin_centers = np.histogram(evec,50)
                X = bin_centers[:-1] + np.diff(bin_centers)/2
            
                C = np.cumsum(N[::-1])
                bins = X[::-1]
                k = 0
                while C[k] < Nfsec and k < 50:
                    k = k+1
    
                thresh = bins[k]
                selecte = evec > thresh
      
                while np.sum(selecte) > Nfsec:
                    thresh = thresh*1.2
                    selecte = evec > thresh
            	
                while np.sum(selecte) < Nfsec:
                    thresh = thresh*0.9
                    selecte = evec > thresh
                
                select = np.append(select, selindsec[selecte])
                    
        Fout = {'ivec':Fivec[select], 'jvec':Fjvec[select], 'evec':Fevec[select],
                'sivec':Fsivec[select], 'sjvec':Fsjvec[select], 'svec':Fsvec[select]}
        Fout['Nfeats']=len(Fout['ivec'])
        #Fout['thresh'] = thresh
        return Fout
          
    def z_jet_p2(self,ImgPyramid,F):
        '''
        Local jet of order three of interest points i,j
        '''  
        feats = len(F['ivec'])        
        Fsvec = F['svec']
        Fsivec = F['sivec']
        Fsjvec = F['sjvec']
        images = ImgPyramid['images']
        
        JAcoeff=[[0 for x in range(self.nmax+1)] for y in range(self.nmax+1)]
        JBcoeff=[[0 for x in range(self.nmax+1)] for y in range(self.nmax+1)]
       
        #initialize
        for n in range(self.nmax+1):
            for m in range(n%2,n+1,2):
                JAcoeff[n][m] = np.zeros(feats,dtype=complex)
                JBcoeff[n][m] = np.zeros(feats,dtype=complex)
        
        for k in range(feats): #(feats+1):
            sk = Fsvec[k] #scale of feature
            i_s = Fsivec[k]
            j_s = Fsjvec[k]
            # window size
            # [size(P(sk).im) is-zrad is+zrad js-zrad js+zrad]
            W = images[sk][i_s-self.zrad:i_s+self.zrad+1,
                           j_s-self.zrad:j_s+self.zrad+1]
            #print(W)
            #print("W shape:",W.shape)
                    
            Wh = W-np.mean(W)
            W = Wh/(np.sum(Wh**2)**0.5)
        
            Wb = images[sk][i_s-self.brad:i_s+self.brad+1,
                            j_s-self.brad:j_s+self.brad+1]
            Wbh = Wb-np.mean(Wb)
            Wb = Wbh/((np.sum(Wbh**2))**0.5)
                        
            for n in range(self.nmax+1):
                for m in range(n%2,n+1,2):
                    JAcoeff[n][m][k] = np.sum(W*self.ZstrucZ[n][m])
                    JBcoeff[n][m][k] = np.sum(Wb*self.BstrucZ[n][m])

        return JAcoeff, JBcoeff

    def zinvariants4(self, JA, JB):
        '''
        oriented invariants
        invariance to affine changes in intensity
        '''
        rows, = JA[0][0].shape
        V = np.zeros((rows, self.ZstrucNdesc))
        A = np.zeros((rows, self.ZstrucNdesc))
        Vb = np.zeros((rows, self.ZstrucNdesc))
        Ab = np.zeros((rows, self.ZstrucNdesc))
        #1 through 7 are oriented gradients, relative to maximum direction
        k = 0
        for n in range(self.nmax+1):
            for m in range(n%2,n+1,2):
                V[:,k] = np.abs(JA[n][m])
                Vb[:,k] = np.abs(JB[n][m])
                A[:,k] = np.angle(JA[n][m])
                Ab[:,k] = np.angle(JB[n][m])
                k = k+1
        V = np.hstack((V, Vb))
        A = np.hstack((A, Ab))
        alpha = np.angle(JA[1][1])
        return V,alpha,A
    
    def detectAndCompute(self, gr_img, mask=None):
        if len(gr_img.shape)!=2:
            raise ValueError("Input image is not a 2D array, possibile non-grayscale")
        P = self.generate_pyramid(gr_img)
        F = self.feat_extract_p2(P)
        Ft = self.feat_thresh_sec(F,*gr_img.shape)
        JA,JB = self.z_jet_p2(P,Ft)
        V,alpha,A = self.zinvariants4(JA, JB)
        kp = [cv2.KeyPoint(x,y,self.zrad*(sc+1)*2,_angle=ang,_response=res,_octave=sc) 
              for x,y,ang,res,sc in zip(Ft['jvec'], Ft['ivec'], np.rad2deg(alpha),
                                        Ft['evec'],Ft['svec'])]
        # Convert zernike descriptors to UINT by dividing with maxdes
        des = np.hstack((np.clip(np.round(V[:, :25]/self.maxdes[0]*255),0,255).astype(np.uint8),
                         np.clip(np.round(V[:,-25:]/self.maxdes[1]*255),0,255).astype(np.uint8)))
        return kp, des#, F, Ft, JA, JB, alpha, A
    
if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057821.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img2 = cv2.imread(path+'data/skerki_small/all/ESC.970622_023824.0546.tif',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE
#img2 = cv2.imread(path+'data/time_lapse_5_cervino_800x600/G0057826.png',1) # iscolor = CV_LOAD_IMAGE_GRAYSCALE

gr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

a = MultiHarrisZernike(Nfeats=600)

import time
st = time.time()
for i in range(1):
    kp, des = a.detectAndCompute(gr1)
print("elapsed: ",(time.time()-st)/1)

outImage	 = cv2.drawKeypoints(gr1, kp, gr1,color=[255,255,0],
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
fig, ax= plt.subplots(dpi=200)
plt.title('Multiscale Harris with Zernike Angles')
plt.axis("off")
plt.imshow(outImage)
plt.show()