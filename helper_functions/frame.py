#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
import numpy as np
np.set_printoptions(precision=5,suppress=True)
import cv2
import time
import logging
from matplotlib import pyplot as plt
from vslam_helper import *

class Frame ():
    '''
 
    '''
    K = np.eye(3)
    D = np.zeros([5,1])
    last_id = -1
    clahe_obj = None
    detector = None
    is_config_set = False
    frlog = logging.getLogger('Frame')
    fig_frame_image = None
    ax1 = None
    ax2 = None
    fig1 = None
    fig2 = None
    
    def __init__(self, image_name, mask_name=None):
        Frame.last_id += 1
        self.frameid = Frame.last_id
        self.mask    = None             # mask for the image
        self.kp_lm_ind = set([])        # Indices of keypoints that already have landmarks
        self.kp_cand_ind = set([])      # Indices of candidate keypoints that dont have landmarks associated
        self.lm_ind = set([])           # Index of landmarks which match kp_lm_ind
        self.T_pnp = np.zeros([4,4])     # Pose in world frame computed from PNP
        self.T_gtsam = np.zeros([4,4])   # Pose in world frame after iSAM2 optimization
        self.kp_m_prev = []
        self.kp_m_next = []

        if not Frame.is_config_set:
            raise ValueError('All required class config not set')
            
        
        Frame.frlog.debug("Pre-processing image: "+image_name)

        pt = time.time()
    
        img = cv2.imread(image_name)
        self.gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        if mask_name is not None:
            self.mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
    
        if Frame.clahe_obj is not None: self.gr = Frame.clahe_obj.apply(self.gr)
        
        self.kp_objs,self.des = Frame.detector.detectAndCompute(self.gr,self.mask)
                
        '''
        if tiling is not None:
            kp = tiled_features(kp, gr.shape, *tiling)
            pbf += " > tiling supression: "+str(len(kp))
        
        if rnm_radius is not None:
            kp = radial_non_max(kp,rnm_radius)
            pbf += " > radial supression: "+str(len(kp))
        '''
        
        self.kp = cv2.KeyPoint_convert(self.kp_objs)
        pbf = "New feature candidates detected: "+str(len(self.kp))
        Frame.frlog.debug("Image Pre-processing time is {:.4f}".format(time.time()-pt))
        Frame.frlog.debug(pbf)

        #self.kp_ud   = cv2.undistortPoints(np.expand_dims(self.kp,1),
        #                                   self.K,self.D)[:,0,:]

    def undistort_matched_pts(self, isForward):
        if isForward:
            kpm = self.kp[self.kp_m_next]
        else:
            kpm = self.kp[self.kp_m_prev]
        return cv2.undistortPoints(np.expand_dims(kpm,1),Frame.K,Frame.D)[:,0,:]

    def show_features(self):
        outImage = cv2.drawKeypoints(self.gr, self.kp_objs, self.gr,color=[255,255,0],
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

        
        plt.title('Multiscale Harris with Zernike Angles')
        plt.axis("off")
        plt.imshow(outImage)
        plt.show()
    
    @staticmethod    
    def draw_point_tracks(fr1,fr2, bool_mask, display_invalid=False, color=(0, 255, 0)):
        '''
        This function extracts takes a 2 images, set of keypoints and a mask of valid
        (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
        The mask should be the same length as matches
        '''
        #bool_mask = mask[:,0].astype(bool)
        
        if len(bool_mask) != len(fr1.kp_m_next):
            raise ValueError('Length of fr1.kp_m_next:{} doesn''t match length of bool_mask:'.format(len(fr1.kp_m_next),len(bool_mask)))
        
        if len(bool_mask) != len(fr2.kp_m_prev):
            raise ValueError('Length of fr2.kp_m_prev:{} doesn''t match length of bool_mask:'.format(len(fr2.kp_m_prev),len(bool_mask)))
        
        left_matches = fr1.kp[fr1.kp_m_next]
        valid_left_matches = left_matches[bool_mask]
        right_matches = fr2.kp[fr2.kp_m_prev]
        valid_right_matches = right_matches[bool_mask]

        img_right_out = cv2.cvtColor(fr2.gr,cv2.COLOR_GRAY2RGB)
    
        thick = round(img_right_out.shape[1]/1000)+1
        for p1,p2 in zip(valid_left_matches,valid_right_matches):
            cv2.arrowedLine(img_right_out, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color=color, thickness=thick)
        return img_right_out
    
    @staticmethod
    def initialize_figures(window_xadj = 65, window_yadj = 430):
        #Frame.fig1, Frame.ax1 = plt.subplots()
        Frame.fig1 = plt.figure(1)
        Frame.ax1 = Frame.fig1.add_subplot(111)
        Frame.ax1.set_title('Image 1 to 2 matches')
        Frame.ax1.axis("off")
        Frame.fig1.subplots_adjust(0,0,1,1)
        plt.get_current_fig_manager().window.setGeometry(window_xadj,window_yadj,640,338)
        
        Frame.fig2 = plt.figure(2)
        Frame.ax2 = Frame.fig2.add_subplot(111, projection='3d')
        Frame.fig2.subplots_adjust(0,0,1,1)
        plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
        Frame.ax2.set_aspect('equal')         # important!
        Frame.fig2.suptitle('Image 1 to 2 after triangulation')
    
    @staticmethod
    def initialize_VSLAM(fr1, fr2, matcher, config_dict):
        Frame.frlog.debug('Length of kp1: {}  Length of kp2: {}'.format(len(fr1.kp),len(fr2.kp)))
    
        matches_12 = knn_match_and_lowe_ratio_filter(matcher, fr1.des, fr2.des, 
                                                     threshold=config_dict['lowe_ratio_test_threshold'])
        Frame.frlog.debug('Length of matches: '+str(len(matches_12)))
        
        '''
        track_output = track_kp_array(kp1, des1, kp2, des2, matches_12)
        kp1_match_12, des1_m, kp2_match_12, des2_m, kp2_cand_pts, des2_cand = track_output
        '''
        Frame.index_from_matches(fr1,fr2,matches_12)
       
        kp1_match_12_ud = fr1.undistort_matched_pts(True)
        kp2_match_12_ud = fr2.undistort_matched_pts(False)
        
        E_12, mask_e_12 = cv2.findEssentialMat(kp1_match_12_ud, 
                                               kp2_match_12_ud,
                                               focal=1.0, pp=(0., 0.), 
                                               method=cv2.RANSAC, **config_dict['findEssential_settings'])
        Frame.frlog.info("Essential matrix: used {} of total {} matches".format(np.sum(mask_e_12),len(kp1_match_12_ud),))
        essen_mat_pts = np.sum(mask_e_12)
        points, R_21, t_21, mask_RP_12 = cv2.recoverPose(E_12, kp1_match_12_ud, kp2_match_12_ud,mask=mask_e_12)
        Frame.frlog.info("Recover pose used {} of total matches in Essential matrix".format(np.sum(mask_RP_12),essen_mat_pts))
        T_2_1 = compose_T(R_21,t_21)
        T_1_2 = T_inv(T_2_1)
        fr2.T_pnp = T_1_2
        Frame.frlog.info("R:\t"+str(R_21).replace('\n','\n\t\t'))
        Frame.frlog.info("t:\t"+str(t_21.T))
            
        img12 = Frame.draw_point_tracks(fr1,fr2,mask_RP_12[:,0].astype(bool), True)
        
        Frame.ax1.imshow(img12)
        plt.draw()
        plt.pause(0.01)
        
        input("Press [enter] to continue.\n")
        
        # Trim the tracked key pts
        '''
        kp1_match_12 = kp1_match_12[mask_RP_12[:,0].astype(bool)]
        kp2_match_12 = kp2_match_12[mask_RP_12[:,0].astype(bool)]
        des2_m = des2_m[mask_RP_12[:,0].astype(bool)]
        
        kp1_match_12_ud = kp1_match_12_ud[mask_RP_12[:,0].astype(bool)]
        kp2_match_12_ud = kp2_match_12_ud[mask_RP_12[:,0].astype(bool)]
        '''
        kp1_match_12, kp2_match_12, des2_m, kp1_match_12_ud, kp2_match_12_ud = trim_using_mask(mask_RP_12,kp1_match_12, 
                                                                   kp2_match_12, des2_m, kp1_match_12_ud, 
                                                                   kp2_match_12_ud)
        
        landmarks_12, mask_tri_12 = triangulate(np.eye(4), T_1_2, kp1_match_12_ud, 
                                                kp2_match_12_ud, None)
        vslog.info("Triangulation used {} of total matches {} matches".format(np.sum(mask_tri_12),len(mask_tri_12)))
    
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        fig2.subplots_adjust(0,0,1,1)
        plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
        ax2.set_aspect('equal')         # important!
        fig2.suptitle('Image 1 to 2 after triangulation')
        
        graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='r')
        set_axes_equal(ax2)
        ax2.view_init(0, -90)
        fig2.canvas.draw_idle(); plt.pause(0.01)
        graph.remove()
        
        if PLOT_LANDMARKS:
            graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='C0')
        '''
        kp2_match_12 = kp2_match_12[mask_tri_12[:,0].astype(bool)]
        des2_m = des2_m[mask_tri_12[:,0].astype(bool)]
        '''
        kp2_match_12, des2_m = trim_using_mask(mask_tri_12, kp2_match_12, des2_m)
        cam_pose_0 = plot_pose3_on_axes(ax2,np.eye(4), axis_length=0.5)
        cam_pose = plot_pose3_on_axes(ax2, T_1_2, axis_length=1.0)
        
        cam_trail_pts = T_1_2[:3,[-1]].T
        cam_pose_trail = plot_3d_points(ax2, cam_trail_pts, linestyle="", color='g', marker=".", markersize=2)
        
        #fig2.canvas.mpl_connect('button_press_event', onClick)
        fig2.canvas.mpl_connect('key_press_event', onKey)
        
        plt.pause(.01)
        input("Press [enter] to continue.\n")
           
        #kp2_pts = remove_redundant_newkps(kp2_pts, kp2_match_12, 5)
        
        vslog.debug("Length of candidate pts: {}".format(len(kp2_cand_pts)))
        #kp2_cand_pts = kp2_cand
        
        img12_newpts = draw_points(img12,kp2_cand_pts, color=[255,255,0])
        fig1_image.set_data(img12_newpts)
        fig1.canvas.draw_idle(); plt.pause(0.01)
        if PAUSES: input("Press [enter] to continue.")
        
    @staticmethod
    def index_from_matches(fr1, fr2, matches):
        '''
   
        '''
        # Go through matches and create list of indices of kp1 and kp2 which matched
        for i,m in enumerate(matches):
            fr1.kp_m_next += [m.queryIdx]
            fr2.kp_m_prev += [m.trainIdx]
        return 