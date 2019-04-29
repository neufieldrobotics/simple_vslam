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
import pickle

class Frame ():
    '''
    Frame objects contain all the variables required to detect, match and propogate
    features and landmarks required in VSLAM.   
    Additionally it contains a number of static functions which operate on multiple
    Frame objects to perform initialization and keypoint propogation.
    We also use class variable to maintain common variables across all Frame objects
    Quick guide to naming convention:
        kp      : KeyPoints
        lm      : Landmarks
        cand    : Candidate pts (which do not have associated landmarks)
        des     : Keypoint Descriptors
        _ind    : Row Indices into the full kp array
        _pt     : Actual points
        _pt_ud  : Actual points undistorted using K and D
    '''
    K = np.eye(3)                         # Camera matrix
    D = np.zeros([5,1])                   # Distortion coefficients
    last_id = -1                          # ID of the last object created  
    clahe_obj = None                      # Contrast Limited Adaptive Histogram Equalization object if being used
    detector = None                       # Feature detector object being used
    matcher = None                        # Matcher object being used
    config_dict = None                    # Dictionary containing all the required configuration settings
    is_config_set = False                 # Flag to make sure user is setting all the required config
    frlog = logging.getLogger('Frame')    # Logger object for console and file logging
    # Visualization
    fig_frame_image = None                # Handle to Frame image which contains the image array being displayed
    ax1 = None                            # Handle to axes in fig 1
    ax2 = None                            # Handle to axes in fig 2
    fig1 = None                           # Handle to fig 1
    fig2 = None                           # Handle to fig 2
    lm_plot_handle = None                 # Handle to plot object for landmarks
    landmarks = None                      # Full array of all 3D landmarks
    
    def __init__(self, image_name, mask_name=None):
        '''
        Frame objects initializer
        '''
        # Populated during initialization
        Frame.last_id += 1  
        self.frame_id = Frame.last_id   # unique id for frame, required for gtsam
        self.mask    = None             # mask for the image
        self.kp_obj = None              # List of keypoint objects calc by Feature Extractor
        self.kp = None                  # Keypoint coordinates as N x 2 array  (flat32)
        self.des = None                 # Feature descriptor array of N x (des_length) (Float32) 

        # Variables forwarded to the next frame
        self.kp_lm_ind = []             # Indices of keypoints that already have landmarks
        self.kp_cand_ind = []           # Indices of candidate keypoints that dont have landmarks associated
        self.lm_ind = []                # Index of landmarks which match kp_lm_ind
        self.T_pnp = np.eye(4)    # Pose in world frame computed from PNP
        self.T_gtsam = np.eye(4)  # Pose in world frame after iSAM2 optimization

        # Variables used in processing frames and also passed to gtsam for optimisation
        self.kp_m_prev_lm_ind = None    # keypoint indices for pre-existing landmarks
        self.kp_m_prev_cand_ind = None  # keypoint indices for new landmarks 
        self.lm_new_ind = None          # landmark indices for new landmakrs
        
        ''''
        kp_j_new_ud  = kp_j_all_ud[-num_cand:]
        
        kp_i_cand_ud = kp_i_cand_ud[mask_RP_cand]
        kp_j_new_ud  = kp_j_new_ud[mask_RP_cand]
        
        lm_if_up = lm_if[mask_RP_lm]
        kpic_match = kpic_match[mask_RP_cand]    
        kpjn_match = kpjn_match[mask_RP_cand]
        desjn_match = desjn_match[mask_RP_cand]
        '''
    

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
    
    def show_features(self):
        '''
        This is a test function to display detected features in the frame object
        '''
        outImage = cv2.drawKeypoints(self.gr, self.kp_objs, self.gr,color=[255,255,0],
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        plt.title('Multiscale Harris with Zernike Angles')
        plt.axis("off")
        plt.imshow(outImage)
        plt.show()
        
    def partition_kp_cand(self):
        '''
        This function operates on the current frame object and partitions its keypoints (kp)
        into candidate points which do not have any associated landmarks
        '''
        kp_ind_set = set(range(len(self.kp)))
        kp_m_prev_cand_ind_set = set(self.kp_m_prev_cand_ind)
        self.kp_cand_ind = np.array(list(kp_ind_set - kp_m_prev_cand_ind_set))
           
    @staticmethod
    def initialize_figures(window_xadj = 65, window_yadj = 430):
        '''
        Initialises two figure windows using Frame's static variables:
            Figure 1 displays the matched features of image i and image j
            Figure 2 displays the pose of the camera and the landmarks of the position
            
        Parameters
        ----------
        window_xadj: widith of figure
        window_yadj : height of figure
        
        Returns
        -------
        None
        
        Modifies / Updates
        -------
        
        '''
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
    def triangulate(T_w_1, T_w_2,  pts_1_2d, pts_2_2d, mask):
        '''
        This function accepts two homogeneous transforms (poses) of 2 cameras in world coordinates,
        along with corresponding matching points and returns the 3D coordinates in world coordinates
        Mask must be a dimensionless array or n, array
        '''
        pts_1 = np.expand_dims(pts_1_2d,1)
        pts_2 = np.expand_dims(pts_2_2d,1)
        T_origin = np.eye(4)
        P_origin = T_origin[:3]
        # calculate the transform of 1 in 2's frame
        T_2_w = T_inv(T_w_2)
        T_2_1 = T_2_w @ T_w_1
        P_2_1 = T_2_1[:3]
        #Frame.frlog.info("P_2_1:\t"+str(P_2_1).replace('\n','\n\t\t\t'))
        
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
                
        # checks the distance threshold after triangulation and create a mask
                if pts_3d_frame1_hom_norm[pt_iter,2]<=0 or \
                   pts_3d_frame1_hom_norm[pt_iter,2]>120: #or Frame.config_dict["Triangulation_settings"]["max_dist"]
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
    def match_and_propagate_keypoints(fr_i, fr_j, initialization=False):
        '''
        This function:
            a. matches for features which were previously tracked and had landmarks
            b. matches for features which are candidates from previous image and 
               not associated with landmark
            
        
        Parameters
        ----------
        fr_i : Frame object
        fr_j : Frame object
        
        Returns
        -------
        None
        
        Modifies / Updates
        -------
        fr_i: kp_lm_ind, lm_ind, kp_cand_ind
        fr_j: kp_m_prev_lm_ind, kp_m_prev_cand_ind
        '''
        
        #if len(kp2) != len(des2):
        #    raise ValueError('Length of kp2:{} doesn''t match length of des2:'.format(len(kp2),len(des2)))
 
        #matches for features which are candidates from previous image and 
        #not associated with landmarks
        if initialization:
            des_i_cand = fr_i.des
        else:            
            des_i_cand = fr_i.des[fr_i.kp_cand_ind]
            
        matches_cand = knn_match_and_lowe_ratio_filter(Frame.matcher, des_i_cand, fr_j.des,
                                                       Frame.config_dict['lowe_ratio_test_threshold'])
        Frame.frlog.debug("Found matches for {} prev candidates out of {}".format(len(matches_cand),len(des_i_cand)))
            
        l_i = []    
        l_j = []    
        for m in matches_cand:
            l_i += [m.queryIdx]
            l_j += [m.trainIdx]
        
        if initialization:
            fr_i.kp_cand_ind = np.array(l_i)
        else:            
            fr_i.kp_cand_ind = fr_i.kp_cand_ind[l_i]
        
        fr_j.kp_m_prev_cand_ind = np.array(l_j)
        
        Frame.frlog.debug("No of fr_i.kp_cand_ind: {}, fr_j.kp_m_prev_cand_ind: {}".format(
                           len(fr_i.kp_cand_ind),len(fr_j.kp_m_prev_cand_ind)))
    
        #matches for features which were previously tracked and had landmarks
        if not initialization:
            des_i_lm = fr_i.des[fr_i.kp_lm_ind]
            
            matches_lm = knn_match_and_lowe_ratio_filter(Frame.matcher, des_i_lm, fr_j.des, 
                                                         Frame.config_dict['lowe_ratio_test_threshold'])
            
            Frame.frlog.debug("Found matches for {} existing landmarks out of {}".format(len(matches_lm),len(des_i_lm)))
            l_i = []    
            l_j = []
            for m in matches_lm:
                l_i += [m.queryIdx] 
                l_j += [m.trainIdx]                 
    
            fr_i.kp_lm_ind = fr_i.kp_lm_ind[l_i]   # l_i_1 is index of matched lm keypoints into fr_i.kp
                                                   # this has to be done this way since des_i_lm is a subset of fr1.des
            fr_i.lm_ind = fr_i.lm_ind[l_i]
            fr_j.kp_m_prev_lm_ind = np.array(l_j)      # fr_j.des is the full list, so m.train gives us index into fr_j.kp
            
            Frame.frlog.debug("No of fr_i.kp_lm_ind: {}, fr_i.lm_ind: {}, fr_j.kp_m_prev_lm_ind: {}".format(
                              len(fr_i.kp_lm_ind),len(fr_i.lm_ind),len(fr_j.kp_m_prev_lm_ind)))

        
        

    @staticmethod
    def combine_and_filter(fr_i, fr_j):
        '''
        Take 2 sets of potentially matching keypoints for landmarks (which have a 3d landmark)
        and candidates (which don't have an associated landmark), stack them into one set
        and run them through a "Esstentially matrix" filter and optionlly "Recover Pose" filter.
        
        Parameters
        ----------
        kp_i_lm : Nx2 array 
            KeyPoint coordinates from frame i which has associated landmarks in lmi
        
        Returns
        ----------

        '''    
        num_landmarks = len(fr_i.kp_lm_ind)
        num_cand = len(fr_j.kp_m_prev_cand_ind)
        
        # COmbine index list of landmarks and candidates
        kp_i_all_ind = np.concatenate((fr_i.kp_lm_ind, fr_i.kp_cand_ind))
        kp_j_all_ind = np.concatenate((fr_j.kp_m_prev_lm_ind, fr_j.kp_m_prev_cand_ind))
                
        # Compute undisorted points from the complete list 
        kp_i_all_ud = cv2.undistortPoints(np.expand_dims(fr_i.kp[kp_i_all_ind],1),Frame.K,Frame.D)[:,0,:]
        kp_j_all_ud = cv2.undistortPoints(np.expand_dims(fr_j.kp[kp_j_all_ind],1),Frame.K,Frame.D)[:,0,:]
                
        # getting the first mask for filter using essential matrix method
        E, mask_e_all = cv2.findEssentialMat(kp_i_all_ud, kp_j_all_ud, 
                                             focal=1.0, pp=(0., 0.), 
                                             method=cv2.RANSAC, **Frame.config_dict['findEssential_settings'])
        essen_mat_pts = np.sum(mask_e_all)  
        
        Frame.frlog.debug("Essential matrix: used {} of total {} matches".format(essen_mat_pts,len(kp_j_all_ud)))
        
        # getting the second mask for filtering using recover pose
        
        if Frame.config_dict['use_RecoverPose_Filter']:
            # Recover Pose filtering is breaking under certain conditions. Leave out for now.
            _, _, _, mask_RP_all = cv2.recoverPose(E, kp_i_all_ud, kp_j_all_ud, mask=mask_e_all)
            Frame.frlog.info("Recover pose: used {} of total {} matches".format(np.sum(mask_RP_all),essen_mat_pts))
        else: 
            mask_RP_all = mask_e_all
        
        # Split the combined mask to lm feature mask and candidate mask
        mask_RP_lm = mask_RP_all[:num_landmarks]
        mask_RP_cand = mask_RP_all[-num_cand:]
        
        # Assign indexes 
        fr_i.kp_lm_ind = fr_i.kp_lm_ind[mask_RP_lm[:,0].astype(bool)]
        fr_j.kp_m_prev_lm_ind = fr_j.kp_m_prev_lm_ind[mask_RP_lm[:,0].astype(bool)]
        
        fr_i.kp_cand_ind = fr_i.kp_cand_ind[mask_RP_cand[:,0].astype(bool)]
        fr_j.kp_m_prev_cand_ind = fr_j.kp_m_prev_cand_ind[mask_RP_cand[:,0].astype(bool)]
        
        fr_i.lm_ind = fr_i.lm_ind[mask_RP_lm[:,0].astype(bool)]

        Frame.frlog.debug("Essential matrix and RP filtered {} landmarks out of {}".format(np.sum(mask_RP_lm),num_landmarks))
        Frame.frlog.debug("Essential matrix and RP filtered {} candidates out of {}".format(np.sum(mask_RP_cand),num_cand))
    
    @staticmethod
    def save_frame(fr_obj, file_name):
        with open(file_name, 'wb') as output:
            pickle.dump(fr_obj, output)
    
    @staticmethod
    def load_frame(file_name):
        with open(file_name, 'rb') as input:
            fr_obj = pickle.load(input)
        return fr_obj
        
    @staticmethod
    def initialize_VSLAM(fr1, fr2):
        '''
        This function takes two Frame objects fr1 and fr2 and initizalizes the algorithm by:
            a. Computing Essential matrix from matched keypoints (kp) fr1 to fr2
            b. Computing pose from Essential matrix between fr1 and fr2
            c. Set the scale for algorithm as unit length between fr1 and fr2
            d. Triangulate landmarks using matched keypoints 
            e. Populates fr2 variables with required information for next frame
        Parameters
        ----------                        
        fr1: Frame object
        fr2: Frame object
        
        Returns
        -------
        None
        '''
        # a. Computing Essential matrix from matched keypoints (kp) fr1 to fr2
        Frame.frlog.debug('Length of kp1: {}  Length of kp2: {}'.format(len(fr1.kp),len(fr2.kp)))

        Frame.match_and_propagate_keypoints(fr1,fr2,initialization=True)
        
        kp1_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr1.kp[fr1.kp_cand_ind],1), Frame.K, Frame.D)[:,0,:]
        kp2_m_prev_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr2.kp[fr2.kp_m_prev_cand_ind],1), Frame.K, Frame.D)[:,0,:]
       
        E_12, mask_e_12 = cv2.findEssentialMat(kp1_cand_pt_ud, 
                                               kp2_m_prev_cand_pt_ud,
                                               focal=1.0, pp=(0., 0.), method=cv2.RANSAC, 
                                               **Frame.config_dict['findEssential_settings'])
        
        Frame.frlog.info("Essential matrix: used {} of total {} matches".format(np.sum(mask_e_12),len(kp2_m_prev_cand_pt_ud)))
        essen_mat_pts = np.sum(mask_e_12)
        
        points, rot_2R1, trans_2t1, mask_RP_12 = cv2.recoverPose(E_12, 
                                                                 kp1_cand_pt_ud, 
                                                                 kp2_m_prev_cand_pt_ud,
                                                                 mask=mask_e_12)
        
        # b. Computing pose from Essential matrix between fr1 and fr2
        # c. Set the scale for algorithm as unit length between fr1 and fr2
        Frame.frlog.info("Recover pose used {} of total matches in Essential matrix {}".format(np.sum(mask_RP_12),essen_mat_pts))
        pose_2T1 = compose_T(rot_2R1,trans_2t1)
        pose_1T2 = T_inv(pose_2T1)
        fr2.T_pnp = pose_1T2
        Frame.frlog.info("1T2 :\t"+str(pose_1T2[:3,:]).replace('\n','\n\t\t'))
                    
        img12 = draw_point_tracks(fr1.kp[fr1.kp_cand_ind], fr2.gr,
                                  fr2.kp[fr2.kp_m_prev_cand_ind],
                                  mask_RP_12[:,0].astype(bool), False, color=[255,255,0])
        
        Frame.fig_frame_image = Frame.ax1.imshow(img12)
        #plt.draw()
        #plt.pause(0.01)
        
        plot_pose3_on_axes(Frame.ax2,np.eye(4), axis_length=0.5)
        Frame.cam_pose = plot_pose3_on_axes(Frame.ax2, pose_1T2, axis_length=1.0)
        
        Frame.cam_trail_pts = pose_1T2[:3,[-1]].T
        Frame.cam_pose_trail = plot_3d_points(Frame.ax2, Frame.cam_trail_pts, linestyle="", color='g', marker=".", markersize=2)
        
        #input("Press [enter] to continue.\n")
        
        fr1.kp_cand_ind         = fr1.kp_cand_ind[mask_RP_12[:,0].astype(bool)]
        fr2.kp_m_prev_cand_ind  = fr2.kp_m_prev_cand_ind[mask_RP_12[:,0].astype(bool)]
        
        # d. Triangulate landmarks using matched keypoints        
        kp1_m_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr1.kp[fr1.kp_cand_ind],1),
                                                             Frame.K, Frame.D)[:,0,:]
        kp2_m_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr2.kp[fr2.kp_m_prev_cand_ind],1),
                                                             Frame.K, Frame.D)[:,0,:]
        
        landmarks_12, mask_tri_12 = Frame.triangulate(np.eye(4), pose_1T2, 
                                                      kp1_m_cand_pt_ud, 
                                                      kp2_m_cand_pt_ud, 
                                                      None)
        Frame.landmarks = landmarks_12
        
        Frame.frlog.info("Triangulation used {} of total matches {} matches".format(np.sum(mask_tri_12),len(mask_tri_12)))
    
        Frame.lm_plot_handle = plot_3d_points(Frame.ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='r')
        set_axes_equal(Frame.ax2)
        Frame.fig2.canvas.draw_idle(); #plt.pause(0.01)
        
        #input("Press [enter] to continue.\n")
        #Frame.lm_plot_handle.remove()

        #kp2_match_12, des2_m = trim_using_mask(mask_tri_12, kp2_match_12, des2_m)
        fr1.kp_cand_ind         = fr1.kp_cand_ind[mask_tri_12[:,0].astype(bool)]
        fr2.kp_m_prev_cand_ind  = fr2.kp_m_prev_cand_ind[mask_tri_12[:,0].astype(bool)]

        fr2.partition_kp_cand()
        Frame.frlog.debug("Length of candidate pts: {}".format(len(fr2.kp_cand_ind)))
        img12_cand = draw_points(img12, fr2.kp[fr2.kp_cand_ind], color=[255,255,0])
        Frame.fig_frame_image.set_data(img12_cand)
        Frame.fig1.canvas.draw_idle(); plt.pause(0.01)
        
        if len(Frame.landmarks) != len(fr2.kp_m_prev_cand_ind)  or len(fr2.kp_m_prev_cand_ind) != len(fr1.kp_cand_ind):
            raise ValueError('Between Frame {} and {}: Length of of kp_m_prev doesnt match kp_m_next or landmarks',format(fr1.frameid,fr2.frameid))

        fr2.lm_new_ind = np.array(range(len(fr2.kp_m_prev_cand_ind))) 
        fr2.lm_ind = fr2.lm_new_ind
        fr2.kp_lm_ind = fr2.kp_m_prev_cand_ind
        
        input("Press [enter] to continue.")  
        
    @staticmethod
    def process_keyframe(fr_i, fr_j):
        '''
        performs the following operations on frame i and frame j
        a. match and propagate keypoints:
         1.matches points with existing landmarks in frame i and frame j and 
           stores their index for respective frames
         2.matches new candidate points to triangulate in frame i and frame j 
           and stores their index for respective frames
        b. combine and filter: combine the above two sets of points and filters 
           them with essential matrix and recover pose method
        c. Slice inlier keypoints, undistort and compute pose using PNP Ransac algorithm
        d. Undistort and Triangulate
        
        Parameters
        ----------                        
        fr_i: Frame object
        fr_j: Frame object
        i = previous frame, j = current frame
        
        Returns
        -------
        None
        
        '''
        Frame.match_and_propagate_keypoints(fr_i, fr_j)
        
        Frame.combine_and_filter(fr_i, fr_j)
        
        time_start = time.time()
            
        # Display translucent mask on image.
        # if mask_j is not None:
        #    gr_j_masked = cv2.addWeighted(mask_j, 0.2, gr_j, 1 - 0.2, 0)
        # else: gr_j_masked = gr_j

        # Display landmarks carried from previous frame        
        img_track_lm = draw_point_tracks(fr_i.kp[fr_i.kp_lm_ind],
                                         fr_j.gr,
                                         fr_j.kp[fr_j.kp_m_prev_lm_ind],
                                         None, False)
        
        # Display candidates carred from previous frame
        img_track_all = draw_point_tracks(fr_i.kp[fr_i.kp_cand_ind],
                                          img_track_lm,
                                          fr_j.kp[fr_j.kp_m_prev_cand_ind],
                                          None, False, color=[255,255,0])
        
        Frame.fig_frame_image.set_data(img_track_all)
        #Frame.fig1.canvas.draw_idle(); plt.pause(0.01)
        #input("Enter to continue")
    
        Frame.frlog.debug("Time elapsed in drawing tracks: {:.4f}".format(time.time()-time_start))
        time_start = time.time()
    
        # success, T_j, mask_PNP = T_from_PNP(lm_if_up, kpjl_match, K, D)
        
        # Slice inlier keypoints from fr_j.kp and undistort
        kp_j_m_lm_pt_ud = cv2.undistortPoints(np.expand_dims(fr_j.kp[fr_j.kp_m_prev_lm_ind],1),
                                                             Frame.K, Frame.D)[:,0,:]
        success, fr_j.T_pnp, mask_pnp = T_from_PNP_norm(Frame.landmarks[fr_i.lm_ind], 
                                                        kp_j_m_lm_pt_ud, 
                                                        **Frame.config_dict['solvePnPRansac_settings']) # repErr = ceil2MSD(1/fr_j.gr.shape[1])
        
        if not success:
            Frame.frlog.critical("PNP failed in frame {}. Exiting...".format(frame_no+1))
            exit()
            
        Frame.frlog.info("PNP inliers: {}  of {}".format(np.sum(mask_pnp),len(fr_i.lm_ind)))
        Frame.frlog.info("PNP_Pose:\t"+str(fr_j.T_pnp[:3,:]).replace('\n','\n\t\t\t'))
        
        plot_pose3_on_axes(Frame.ax2, fr_j.T_pnp, axis_length=2.0, center_plot=True, line_obj_list=Frame.cam_pose)
        
        Frame.cam_trail_pts = np.append(Frame.cam_trail_pts,fr_j.T_pnp[:3,[-1]].T,axis=0)
        plot_3d_points(Frame.ax2,Frame.cam_trail_pts , line_obj=Frame.cam_pose_trail, linestyle="", color='g', marker=".", markersize=2)
        #Frame.fig2.canvas.draw_idle(); #plt.pause(0.01)
        #input("Press [enter] to continue.")
                    
        fr_i.lm_ind = fr_i.lm_ind[mask_pnp[:,0].astype(bool)] 
        fr_i.kp_lm_ind = fr_i.kp_lm_ind[mask_pnp[:,0].astype(bool)] 
        fr_j.kp_m_prev_lm_ind = fr_j.kp_m_prev_lm_ind[mask_pnp[:,0].astype(bool)] 
        Frame.frlog.debug("Time elapsed in PNP: {:.4f}".format(time.time()-time_start))
        time_start = time.time()
        
        # Undistort and Triangulate
        kp_i_m_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr_i.kp[fr_i.kp_cand_ind],1),
                                                             Frame.K, Frame.D)[:,0,:]
        kp_j_m_cand_pt_ud = cv2.undistortPoints(np.expand_dims(fr_j.kp[fr_j.kp_m_prev_cand_ind],1),
                                                             Frame.K, Frame.D)[:,0,:]
        
        lm_j_new, mask_tri = Frame.triangulate(fr_i.T_pnp, fr_j.T_pnp, 
                                           kp_i_m_cand_pt_ud, 
                                           kp_j_m_cand_pt_ud, 
                                           None)
        Frame.frlog.debug("Time elapsed in triangulate: {:.4f}".format(time.time()-time_start)) 
        time_start = time.time()
    
        Frame.frlog.debug("Points retained after triangulation: {} out of length: {}".format(np.sum(mask_tri),len(mask_tri)))
        
        #if len(kp_prev_cand)>0:
        img_rej_pts = draw_point_tracks(fr_i.kp[fr_i.kp_cand_ind], 
                                        img_track_all, 
                                        fr_j.kp[fr_j.kp_m_prev_cand_ind], 
                                        (1-mask_tri)[:,0].astype(bool), True, color=[255,0,0])
        #else: img_rej_pts = img_track_all
        Frame.fig_frame_image.set_data(img_rej_pts)
        #Frame.fig1.canvas.draw_idle(); #plt.pause(0.01)
        #input("Enter to continue")

        Frame.frlog.debug("Time elapsed in draw pt tracks: {:.4f} ".format(time.time()-time_start)) 
        time_start = time.time()
    
        fr_i.kp_cand_ind = fr_i.kp_cand_ind[mask_tri[:,0].astype(bool)]
        fr_j.kp_m_prev_cand_ind = fr_j.kp_m_prev_cand_ind[mask_tri[:,0].astype(bool)]
        
        #try:    
        #    Frame.lm_plot_handle.remove()
        #except NameError:
        #    pass
        
        #print(lm_j_new)
        
        plot_3d_points(Frame.ax2, lm_j_new, line_obj=Frame.lm_plot_handle, 
                       linestyle="", color='g', marker=".", markersize=2)
        
        Frame.fig2.canvas.draw_idle(); #plt.pause(0.01)
        #input("Enter to continue")

    
        # Remove pose lines and landmarks to speed plotting
        #if PLOT_LANDMARKS:
        #    graph_newlm = plot_3d_points(ax2, lm_j, linestyle="", color='C0', marker=".", markersize=2)    
        #    fig2.canvas.draw_idle(); #plt.pause(0.01)
        
        num_curr_landmarks = len(Frame.landmarks)
        num_new_landmarks = len(lm_j_new)
        fr_j.lm_new_ind = np.array(range(num_curr_landmarks,num_curr_landmarks+num_new_landmarks))
        fr_i.kp_lm_ind = np.concatenate((fr_i.kp_lm_ind, fr_i.kp_cand_ind))
        fr_j.kp_lm_ind = np.concatenate((fr_j.kp_m_prev_lm_ind, fr_j.kp_m_prev_cand_ind))
        fr_j.lm_ind = np.concatenate((fr_i.lm_ind, fr_j.lm_new_ind))
        #print('previous indexes:',fr_j.kp_m_prev_lm_ind)                             
        Frame.landmarks = np.vstack((Frame.landmarks, lm_j_new))
       
        #print("length of landmark array:",len(Frame.landmarks))
        #print("length of lm_ind:",len(fr_j.lm_ind))
        # partition kp_m into two sets
        
        fr_j.partition_kp_cand()
        Frame.frlog.debug(" {}".format(len(fr_j.kp_cand_ind)))
        img_cand_pts = draw_points(img_rej_pts,fr_j.kp[fr_j.kp_cand_ind], 
                                  color=[255,255,0])
        Frame.fig_frame_image.set_data(img_cand_pts)

        Frame.fig1.canvas.draw_idle(); #plt.pause(0.01)
                      
        Frame.fig1.canvas.start_event_loop(0.001)
        Frame.fig2.canvas.start_event_loop(0.001)

        #frame_no += 1
            
        #Frame.frlog.debug("FRAME seq {} COMPLETE".format(str(frame_no)))
    
        #return gr_j, kpjl, desjl, kp_j_cand, des_j_cand, lm_j_up, T_j
    
'''    
    @staticmethod
    def extract_and_store_kp(fr,):
'''    