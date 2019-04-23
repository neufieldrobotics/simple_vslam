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
    graph= None
    landmarks = None    
    
    def __init__(self, image_name, mask_name=None):
        Frame.last_id += 1
        self.frameid = Frame.last_id
        self.mask    = None              # mask for the image
        self.kp_lm_ind = []              # Indices of keypoints that already have landmarks
        self.kp_cand_ind = []            # Indices of candidate keypoints that dont have landmarks associated
        self.kp_cand_pt = np.array([])
        self.lm_ind = []                 # Index of landmarks which match kp_lm_ind
        self.T_pnp = np.zeros([4,4])     # Pose in world frame computed from PNP
        self.T_gtsam = np.zeros([4,4])   # Pose in world frame after iSAM2 optimization
        self.kp_m_prev_ind = []
        self.kp_m_next_ind = []
        self.kp_m_next_pt = None         # Keypoints matched with next frame
        self.kp_m_prev_pt = None         # Keypoints matched with prev frame
        self.kp_m_next_pt_ud = None      # Undistorted Keypoints matched with next frame
        self.kp_m_prev_pt_ud = None      # Undistorted Keypoints matched with prev frame

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
    '''
    def undistort_matched_pts(self, isForward):
        if isForward:
            kpm = self.kp[self.kp_m_next_ind]
        else:
            kpm = self.kp[self.kp_m_prev_ind]
        return cv2.undistortPoints(np.expand_dims(kpm,1),Frame.K,Frame.D)[:,0,:]
    '''
    
    def show_features(self):
        outImage = cv2.drawKeypoints(self.gr, self.kp_objs, self.gr,color=[255,255,0],
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

        
        plt.title('Multiscale Harris with Zernike Angles')
        plt.axis("off")
        plt.imshow(outImage)
        plt.show()
        
    def partition_kp_cand(self):
        kp_ind_set = set(range(len(self.kp)))
        kp_m_prev_ind_set = set(self.kp_m_prev_ind)
        self.kp_cand_ind = list(kp_ind_set - kp_m_prev_ind_set)
        self.kp_cand_pt = self.kp[self.kp_cand_ind]
    
    @staticmethod    
    def draw_point_tracks(fr1,fr2, bool_mask, display_invalid=False, color=(0, 255, 0)):
        '''
        This function extracts takes a 2 images, set of keypoints and a mask of valid
        (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
        The mask should be the same length as matches
        '''
        #bool_mask = mask[:,0].astype(bool)
        
        if len(bool_mask) != len(fr1.kp_m_next_ind):
            raise ValueError('Length of fr1.kp_m_next:{} doesn''t match length of bool_mask:'.format(len(fr1.kp_m_next_ind),len(bool_mask)))
        
        if len(bool_mask) != len(fr2.kp_m_prev_ind):
            raise ValueError('Length of fr2.kp_m_prev:{} doesn''t match length of bool_mask:'.format(len(fr2.kp_m_prev_ind),len(bool_mask)))
        
        left_matches = fr1.kp[fr1.kp_m_next_ind]
        valid_left_matches = left_matches[bool_mask]
        right_matches = fr2.kp[fr2.kp_m_prev_ind]
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
        Frame.ax2.view_init(0, -90)
        
    @staticmethod
    def trim_using_masks(mask, fr1, fr2):
        fr1.kp_m_next_ind   = fr1.kp_m_next_ind[mask]
        fr1.kp_m_next_pt    = fr1.kp_m_next_pt[mask]
        fr1.kp_m_next_pt_ud = fr1.kp_m_next_pt_ud[mask]
        fr2.kp_m_prev_ind   = fr2.kp_m_prev_ind[mask]
        fr2.kp_m_prev_pt    = fr2.kp_m_prev_pt[mask]
        fr2.kp_m_prev_pt_ud = fr2.kp_m_prev_pt_ud[mask]
        
    @staticmethod
    def triangulate(T_w_1, T_w_2, fr1, fr2, mask):
        '''
        This function accepts two homogeneous transforms (poses) of 2 cameras in world coordinates,
        along with corresponding matching points and returns the 3D coordinates in world coordinates
        Mask must be a dimensionless array or n, array
        '''
        pts_1 = np.expand_dims(fr1.kp_m_next_pt_ud,1)
        pts_2 = np.expand_dims(fr2.kp_m_prev_pt_ud,1)
        T_origin = np.eye(4)
        P_origin = T_origin[:3]
        # calculate the transform of 1 in 2's frame
        T_2_w = T_inv(T_w_2)
        T_2_1 = T_2_w @ T_w_1
        P_2_1 = T_2_1[:3]
        Frame.frlog.info("P_2_1:\t"+str(P_2_1).replace('\n','\n\t\t\t'))
        
        t_2_1 = T_2_1[:3,-1]
        Frame.frlog.debug("No of points in pts_1: {}".format(pts_1.shape))
        # Calculate points in 0,0,0 frame
        if mask is None:
            pts_3d_frame1_hom = cv2.triangulatePoints(P_origin, P_2_1, pts_1, pts_2).T
            mask = np.ones((pts_1.shape[0],1),dtype='uint8')
        else:
            pts_3d_frame1_hom = cv2.triangulatePoints(P_origin, P_2_1, pts_1[mask==1], 
                                                  pts_2[mask==1]).T
        pts_3d_frame1_hom_norm = pts_3d_frame1_hom /  pts_3d_frame1_hom[:,-1][:,None]
        
        pt_iter = 0
        rows_to_del = []
        ANGLE_THRESHOLD = np.radians(.5)    
        
        for i,v in enumerate(mask):
            if v==1:
                lm_cand = pts_3d_frame1_hom_norm[pt_iter,:3]
                p1 = pts_1[i,0,:]
                p2 = pts_2[i,0,:]
                pdist = np.linalg.norm(p2-p1)
                #angle = angle_between(lm_cand,lm_cand-t_2_1)
                #dist = np.linalg.norm(lm_cand-t_2_1)
    
                if pts_3d_frame1_hom_norm[pt_iter,2]<=0 or \
                   pts_3d_frame1_hom_norm[pt_iter,2]>100: #or \
                   #pdist < .01: 
                    #dist > 50.0:
                    #angle < ANGLE_THRESHOLD:
                    #print ("Point is negative")
                    mask[i,0]=0 
                    rows_to_del.append(pt_iter)
                pt_iter +=1
        
        pts_3d_frame1_hom_norm = np.delete(pts_3d_frame1_hom_norm,rows_to_del,axis=0)
        # Move 3d points to world frame by transforming with T_w_1
        pts_3d_w_hom = pts_3d_frame1_hom_norm @ T_w_1.T
        pts_3d_w = pts_3d_w_hom[:, :3]
        return pts_3d_w, mask        
    
    @staticmethod
    def initialize_VSLAM(fr1, fr2, matcher, config_dict):
        Frame.frlog.debug('Length of kp1: {}  Length of kp2: {}'.format(len(fr1.kp),len(fr2.kp)))
    
        matches_12 = knn_match_and_lowe_ratio_filter(matcher, fr1.des, fr2.des, 
                                                     threshold=config_dict['lowe_ratio_test_threshold'])
        Frame.frlog.debug('Length of matches: '+str(len(matches_12)))
        
        Frame.index_from_matches(fr1,fr2,matches_12)
       
        E_12, mask_e_12 = cv2.findEssentialMat(fr1.kp_m_next_pt_ud, 
                                               fr2.kp_m_prev_pt_ud,
                                               focal=1.0, pp=(0., 0.), 
                                               method=cv2.RANSAC, **config_dict['findEssential_settings'])
        
        Frame.frlog.info("Essential matrix: used {} of total {} matches".format(np.sum(mask_e_12),len(fr1.kp_m_next_pt_ud)))
        essen_mat_pts = np.sum(mask_e_12)
        
        points, rot_2R1, trans_2t1, mask_RP_12 = cv2.recoverPose(E_12, 
                                                                 fr1.kp_m_next_pt_ud, 
                                                                 fr2.kp_m_prev_pt_ud,
                                                                 mask=mask_e_12)
        print("maskrp12: ", mask_RP_12.dtype, mask_RP_12.shape)
        
        Frame.frlog.info("Recover pose used {} of total matches in Essential matrix {}".format(np.sum(mask_RP_12),essen_mat_pts))
        pose_2T1 = compose_T(rot_2R1,trans_2t1)
        pose_1T2 = T_inv(pose_2T1)
        fr2.T_pnp = pose_1T2
        Frame.frlog.info("R:\t"+str(pose_1T2[:3,:3]).replace('\n','\n\t\t'))
        Frame.frlog.info("t:\t"+str(pose_1T2[:3,-1].T))
            
        img12 = Frame.draw_point_tracks(fr1,fr2,mask_RP_12[:,0].astype(bool), True)
        
        Frame.fig_frame_image = Frame.ax1.imshow(img12)
        plt.draw()
        plt.pause(0.01)
        
        plot_pose3_on_axes(Frame.ax2,np.eye(4), axis_length=0.5)
        Frame.cam_pose = plot_pose3_on_axes(Frame.ax2, pose_1T2, axis_length=1.0)
        
        Frame.cam_trail_pts = pose_1T2[:3,[-1]].T
        Frame.cam_pose_trail = plot_3d_points(Frame.ax2, Frame.cam_trail_pts, linestyle="", color='g', marker=".", markersize=2)
        
        input("Press [enter] to continue.\n")
        
        Frame.trim_using_masks(mask_RP_12[:,0].astype(bool), fr1, fr2)
        
        landmarks_12, mask_tri_12 = Frame.triangulate(np.eye(4), pose_1T2, fr1, fr2, None)
        Frame.landmarks = landmarks_12
        
        Frame.frlog.info("Triangulation used {} of total matches {} matches".format(np.sum(mask_tri_12),len(mask_tri_12)))
    
        #if PLOT_LANDMARKS:
        Frame.graph = plot_3d_points(Frame.ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='r')
        set_axes_equal(Frame.ax2)
        Frame.fig2.canvas.draw_idle(); plt.pause(0.01)
        
        input("Press [enter] to continue.\n")
        Frame.graph.remove()

        #kp2_match_12, des2_m = trim_using_mask(mask_tri_12, kp2_match_12, des2_m)
        Frame.trim_using_masks(mask_tri_12[:,0].astype(bool), fr1, fr2)

        fr2.partition_kp_cand()
        Frame.frlog.debug("Length of candidate pts: {}".format(len(fr2.kp_cand_ind)))
        img12_cand = draw_points(img12, fr2.kp_cand_pt, color=[255,255,0])
        Frame.fig_frame_image.set_data(img12_cand)
        Frame.fig1.canvas.draw_idle(); plt.pause(0.01)
        
        if len(Frame.landmarks) != len(fr2.kp_m_prev_ind)  or len(fr2.kp_m_prev_ind) != len(fr1.kp_m_next_ind):
            raise ValueError('Between Frame {} and {}: Length of of kp_m_prev doesnt match kp_m_next or landmarks',format(fr1.frameid,fr2.frameid))
        #if PAUSES: 
        input("Press [enter] to continue.")
        
    @staticmethod
    def index_from_matches(fr1, fr2, matches):
        '''
   
        '''
        fr1_kp_m_next_ind = []
        fr2_kp_m_prev_ind = []
        # Go through matches and create list of indices of kp1 and kp2 which matched
        for i,m in enumerate(matches):
            fr1_kp_m_next_ind += [m.queryIdx]
            fr2_kp_m_prev_ind += [m.trainIdx]
            
        fr1.kp_m_next_ind = np.array(fr1_kp_m_next_ind)
        fr2.kp_m_prev_ind = np.array(fr2_kp_m_prev_ind)
        
        fr1.kp_m_next_pt = fr1.kp[fr1.kp_m_next_ind]
        fr2.kp_m_prev_pt = fr2.kp[fr2.kp_m_prev_ind]
        
        fr1.kp_m_next_pt_ud = cv2.undistortPoints(np.expand_dims(fr1.kp_m_next_pt,1),Frame.K,Frame.D)[:,0,:]
        fr2.kp_m_prev_pt_ud = cv2.undistortPoints(np.expand_dims(fr2.kp_m_prev_pt,1),Frame.K,Frame.D)[:,0,:]
               
        return 
    
    
'''    
    @staticmethod
    def extract_and_store_kp(fr,):
'''    