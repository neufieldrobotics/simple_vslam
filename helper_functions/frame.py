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
from colorama import Fore, Style
import os, sys

class landmark():
    def __init__(self, fr_i_id,fr_j_id, kp_i_pt, kp_j_pt, coord_3d):
        '''
        landmark obj initializer
        '''
        # Populated during initialization
        self.observed_kps = {fr_i_id:kp_i_pt, fr_j_id:kp_j_pt}
        self.coord_3d = coord_3d

    def add_observation(self,fr_id, kp_j_pt):
        '''
        Add new observation of existing landmark
        '''
        self.observed_kps[fr_id] = kp_j_pt

    def keypoint_in_frame(self,fr_id):
        '''
        Return observation in given frame
        '''
        return self.observed_kps[fr_id]
    def __str__(self):
        return "coords: "+ str(self.coord_3d[0,:]) + "\nObservations:".format(self.coord_3d) + str(self.observed_kps)

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
    ![alternate text](docs/frame_data_flow.png)
    '''
    K = np.eye(3)                         # Camera matrix
    D = np.zeros([5,1])                   # Distortion coefficients
    last_id = -1                          # ID of the last object created
    clahe_obj = None                      # Contrast Limited Adaptive Histogram Equalization object if being used
    detector = None                       # Feature detector object being used
    descriptor = None                     # Feature detector object being used
    matcher = None                        # Matcher object being used
    config_dict = {}                      # Dictionary containing all the required configuration settings
    is_config_set = False                 # Flag to make sure user is setting all the required config
    groundtruth_pose_dict = None                   # ground truth pose dictionary

    # Visualization
    fig_frame_image = None                # Handle to Frame image which contains the image array being displayed
    ax1 = None                            # Handle to axes in fig 1
    ax2 = None                            # Handle to axes in fig 2
    fig1 = None                           # Handle to fig 1
    fig2 = None                           # Handle to fig 2
    new_lm_plot_handle = None             # Handle to plot object for landmarks
    all_lm_plot_handle = None             # Handle to plot object for landmarks
    unused_lm_plot_handle = None          # Handle to plot object for landmarks
    landmarks = None                      # Full array of all 3D landmarks
    lm_obs_count = None                   # Number of times a landmark has been observed
    landmark_array = np.array([])         # NP Array of landmark objects
    ax2_plot_limits = np.zeros((3,2))     # Variable to track the current extent of the plot

    frlog = logging.getLogger('Frame')    # Logger object for console and file logging

    def __init__(self, image_name, mask_name=None):
        '''
        Frame objects initializer
        '''
        # Populated during initialization
        Frame.last_id += 1
        self.image_name = os.path.splitext(os.path.basename(image_name))[0]
        self.frame_id = Frame.last_id   # unique id for frame, required for gtsam
        self.mask    = None             # mask for the image
        self.kp_obj = None              # List of keypoint objects calc by Feature Extractor
        self.kp = None                  # Keypoint coordinates as N x 2 array  (floatt32)
        self.kp_ud = None               # Keypoint undistorted coordinates as N x 2 array  (flat32) (NOT NOMALIZED)
        self.kp_ud_norm = None          # Keypoint normalized undistorted coordinates as N x 2 array  (flat32)
        self.des = None                 # Feature descriptor array of N x (des_length) (Float32)

        # Variables forwarded to the next frame
        self.kp_lm_ind = []             # Indices of keypoints that already have landmarks
        self.kp_lm_ind_non_match = []   # Indices of keypoints with landmarks that could not be matched with next image using matcher
        self.kp_cand_ind = []           # Indices of candidate keypoints that dont have landmarks associated
        self.lm_ind = []                # Index of landmarks which match kp_lm_ind
        self.lm_ind_non_match = []      # Index of landmarks which correspond to kp_lm_ind_non_match
        self.T_pnp = np.eye(4)          # Pose in world frame computed from PNP
        self.T_gtsam = np.eye(4)        # Pose in world frame after iSAM2 optimization
        self.T_groundtruth = None  # Ground truth poses read from file

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

        Frame.frlog.info("Pre-processing image: "+image_name)

        pt = time.time()

        img = cv2.imread(image_name)
        self.gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if mask_name is not None:
            self.mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)

        if Frame.clahe_obj is not None: self.gr = Frame.clahe_obj.apply(self.gr)

        if Frame.detector == Frame.descriptor:
            self.kp_objs, self.des = Frame.detector.detectAndCompute(self.gr, self.mask)
        else:
            detector_kp_objs = Frame.detector.detect(self.gr,self.mask)

            #detector_kp_objs = radial_non_max(detector_kp_objs,self.gr.shape, kernel_size=(3,3))

            #detector_kp_objs = tiled_features(detector_kp_objs, self.gr.shape, 6, 2, no_features=2400 )


            '''
            if rnm_radius is not None:
                kp = radial_non_max(kp,rnm_radius)
                pbf += " > radial supression: "+str(len(kp))

            if tiling is not None:
                kp = tiled_features(kp, gr.shape, *tiling)
                pbf += " > tiling supression: "+str(len(kp))


            # Display translucent mask on image.
            # if mask_j is not None:
            #    gr_j_masked = cv2.addWeighted(mask_j, 0.2, gr_j, 1 - 0.2, 0)
            # else: gr_j_masked = gr_j

            '''

            self.kp_objs, self.des = Frame.descriptor.compute(self.gr, detector_kp_objs)

        self.kp = cv2.KeyPoint_convert(self.kp_objs)
        self.kp_ud = cv2.undistortPoints(np.expand_dims(self.kp,1),
                                         Frame.K,
                                         Frame.D,
                                         P=np.hstack((Frame.K,np.zeros((3,1)))))[:,0,:]
        self.kp_ud_norm = cv2.undistortPoints(np.expand_dims(self.kp,1),
                                              Frame.K,
                                              Frame.D)[:,0,:]
        pbf = "New feature candidates detected: " + str(len(self.kp))

        if Frame.groundtruth_pose_dict:
            self.T_groundtruth = Frame.groundtruth_pose_dict[self.image_name]

        Frame.frlog.debug("Image Pre-processing time is {:.4f}".format(time.time()-pt))
        Frame.frlog.debug(pbf)

    def __repr__(self):
        st = ("FrameId: \t{}\nImage: \t\t{}".format(self.frame_id, self.image_name) +
             "\nPNP_Pose:\t"+str(self.T_pnp[:3,:]).replace('\n','\n\t\t\t') +
             "\nGTSAM_Pose:\t"+str(self.T_gtsam[:3,:]).replace('\n','\n\t\t\t'))
        return st

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
    def update_plot_limits(pts=None):
        '''
        Return the required 3d plot limits to show all data
        '''
        if pts is None:
            pts = np.array([ Frame.ax2.get_xlim3d(),
                             Frame.ax2.get_ylim3d(),
                             Frame.ax2.get_zlim3d()  ])
        else:
            pts = pts.T
        all_pts = np.hstack((Frame.ax2_plot_limits, pts))
        Frame.ax2_plot_limits = np.hstack((np.min(all_pts,axis=1)[:,None], np.max(all_pts,axis=1)[:,None]))


    @staticmethod
    def initialize_figures(window_xadj = 65, window_yadj = 430):
        '''
        Initialises two figure windows using Frame's static variables:
            Figure 1 displays the matched features of image i and image j
            Figure 2 displays the pose of the camera and the landmarks of the position

        Parameters
        ----------
        window_xadj: width of figure
        window_yadj: height of figure

        Returns
        -------
        None

        Modifies / Updates
        -------
        '''
        #Frame.fig1, Frame.ax1 = plt.subplots()
        Frame.fig1 = plt.figure(1)
        Frame.ax1 = Frame.fig1.add_subplot(111)
        Frame.ax1.set_title('Frame 1')
        Frame.ax1.axis("off")
        Frame.fig1.subplots_adjust(0,0,1,1)
        plt.get_current_fig_manager().window.setGeometry(window_xadj,window_yadj,1036,842)

        Frame.fig2 = plt.figure(2)
        Frame.ax2 = Frame.fig2.add_subplot(111, projection='3d')
        Frame.fig2.subplots_adjust(0,0,1,1)
        plt.get_current_fig_manager().window.setGeometry(1036+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
        if sys.version_info[:2] == (3, 5):
            Frame.ax2.set_aspect('equal')         # important!

        Frame.fig2.suptitle('Frame 1')
        plt.figtext(0.1,0.05, "Press: s - Toggle Pause, p - Toggle Pause after every frame, q - quit", figure = Frame.fig2)
        Frame.ax2.view_init(0, -90)


    @staticmethod
    def triangulate(Pose_wT1, Pose_wT2,  pts_1_2d, pts_2_2d, mask):
        '''
        This function accepts two homogeneous transforms (poses) of 2 cameras in world coordinates,
        along with corresponding matching points and returns the 3D coordinates in world coordinates.
        Additionally, the triangulated points are filtered based on various critereia to avoid points
        which would have poor triangulation.

        Parameters
        ----------
        Pose_wT1: 4x4 array Homogenous Transform of Camera 1 pose in world coordinates
        Pose_wT2: 4x4 array Homogenous Transform of Camera 1 pose in world coordinates
        pts_1_2d: Nx2 array of Undistorted and normalized keypoints in camera 1
        pts_2_2d: Nx2 array of Undistorted and normalized keypoints in camera 2
        mask: Nx1 Mask of 1s and 0s with 1s for points which should be triangulated

        Returns
        -------
        pts_3d_w: Nx3 array of Keypoint location in world coordinates
        mask: Nx1 array with 1s for triangulation inliers
        '''
        Frame.frlog.info("wT1 :\t"+str(Pose_wT1[:3,:]).replace('\n','\n\t\t'))
        Frame.frlog.info("wT2 :\t"+str(Pose_wT2[:3,:]).replace('\n','\n\t\t'))

        # Convert points to 3D array for processing
        pts_1 = np.expand_dims(pts_1_2d,1)
        pts_2 = np.expand_dims(pts_2_2d,1)

        # Invert poses to find Transform from cam1 and cam2 to world frame
        Pose_1Tw = T_inv(Pose_wT1)
        Pose_2Tw = T_inv(Pose_wT2)

        # Extract Projection matrices from Transform. Usually P1 = 1Tw * K, but in our
        # case K = np.eye(3) since the points are normalized
        Proj_1Tw = Pose_1Tw[:3]
        Proj_2Tw = Pose_2Tw[:3]
        #Frame.frlog.debug("Proj_1Tw:\t"+str(Proj_1Tw).replace('\n','\n\t\t\t'))
        #Frame.frlog.debug("Proj_2Tw:\t"+str(Proj_2Tw).replace('\n','\n\t\t\t'))

        #Pose_1T2 = Pose_1Tw @ Pose_wT2
        #trans_1T2 = Pose_1T2[:3,-1]

        trans_wT1 = Pose_wT1[:3,-1]
        trans_wT2 = Pose_wT2[:3,-1]

        if mask is None:
            pts_3d_world_hom = cv2.triangulatePoints(Proj_1Tw, Proj_2Tw, pts_1, pts_2).T
            mask = np.ones((pts_1.shape[0],1),dtype='uint8')
        else:
            pts_3d_world_hom = cv2.triangulatePoints(Proj_1Tw, Proj_2Tw, pts_1[mask==1],
                                                     pts_2[mask==1]).T
        # pts_3d_world_hom is a Nx4 array of non-nomalized homogenous coorinates
        # So we normalize them by dividing with last element of each row
        # [:, None] returns the last elements of each row as a Nx1 2D instead of a 1
        pts_3d_world_hom_norm = pts_3d_world_hom /  pts_3d_world_hom[:,-1][:,None]

        pt_iter = 0
        rows_to_del = []

        ANGLE_THRESHOLD = np.deg2rad(Frame.config_dict["Triangulation_settings"]["subtended_angle_threshold"])
        Z_THRESHOLD = Frame.config_dict["Triangulation_settings"]["z_threshold"] #* Frame.scale


        for i,mask_val in enumerate(mask):
            if mask_val==1:
                # Find lm candidate point in 2nd Cameras frame
                # Homogenous lm candidate in world frame
                pt_wX_hom = pts_3d_world_hom_norm[pt_iter]
                # lm candidate in frame 2
                pt_2X = (pt_wX_hom @ Pose_2Tw.T )[:3]

                # Find angle between vectors from 2 Poses to the lm candidate point
                pt_wX = pt_wX_hom[:3]
                beta = angle_between(pt_wX-trans_wT1,pt_wX-trans_wT2)

                # Calculate point parallax
                # kp1 = pts_1_2d[i] #pts_1[i,0,:]
                # kp2 = pts_2_2d[i] #pts_2[i,0,:]
                # parallax = np.linalg.norm(kp2-kp1)

                #dist = np.linalg.norm(pt_wX-trans_wT2)

                # Make sure triangulated point is in front of 2nd frame
                # checks the z_threshold threshold after triangulation and create a mask
                # This eliminates points that lie beyond the circle where trans_1T2 is the a cord
                # and the circle represents the locus of all points subtending angle_threshold to the cord

                #print("Pt z:", pt_2X[2])

                if pt_2X[2]<=0 or \
                   pt_2X[2]>Z_THRESHOLD or \
                   beta < ANGLE_THRESHOLD: #or \
                   #parallax > 100 / 800:
                   #dist > 50.0:
                    mask[i,0]=0
                    rows_to_del.append(pt_iter)
                #Frame.frlog.debug("Pt: {}, parallax:{:.3f}, angle: {:.3f}, dist: {:.2f} accepted: {}".format(pt_2X,parallax,np.rad2deg(beta),dist, mask[i,0]))
                pt_iter +=1

        pts_3d_world_hom_norm = np.delete(pts_3d_world_hom_norm,rows_to_del,axis=0)
        pts_3d_world = pts_3d_world_hom_norm[:, :3]
        return pts_3d_world, mask

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
        fr_i: kp_lm_ind, lm_ind, kp_cand_ind, kp_lm_ind_non_match
        fr_j: kp_m_prev_lm_ind, kp_m_prev_cand_ind
        '''
        # matches for features which are candidates from previous image and
        # not associated with landmarks
        if initialization:
            des_i_cand = fr_i.des
            kp_i_cand_pts = fr_i.kp

        else:
            des_i_cand = fr_i.des[fr_i.kp_cand_ind]
            kp_i_cand_pts = fr_i.kp[fr_i.kp_cand_ind]

        print ("\n\nFrame.matcher: ", Frame.matcher)

        pixel_matching_dist = Frame.config_dict.get('pixel_matching_dist')
        if pixel_matching_dist:
            cand_dist_mask, _, fr_j.kp_tree = keypoint_distance_search_mask(kp_i_cand_pts,
                                                                            fr_j.kp ,
                                                                            pixel_matching_dist)
            print("len kp_i_cand_pts {}, frj.kp {}, mask shape {} ".format(len( kp_i_cand_pts),len(fr_j.kp), cand_dist_mask.shape))

        else:
            cand_dist_mask = None

        matches_cand = knn_match_and_lowe_ratio_filter(Frame.matcher, des_i_cand, fr_j.des,
                                                       Frame.config_dict['lowe_ratio_test_threshold'],
                                                       dist_mask_12 = None)
        dbg_str = "Found {} / {} prev candidates".format(len(matches_cand),len(des_i_cand))

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

        #Frame.frlog.debug("No of fr_i.kp_cand_ind: {}, fr_j.kp_m_prev_cand_ind: {}".format(
        #                   len(fr_i.kp_cand_ind),len(fr_j.kp_m_prev_cand_ind)))

        #matches for features which were previously tracked and had landmarks
        if not initialization:
            des_i_lm = fr_i.des[fr_i.kp_lm_ind]
            kp_i_lm_pts = fr_i.kp[fr_i.kp_lm_ind]

            if pixel_matching_dist:
                lm_dist_mask,_ , _ = keypoint_distance_search_mask(kp_i_lm_pts,
                                                                   fr_j.kp ,
                                                                   pixel_matching_dist,
                                                                   kp2_tree = fr_j.kp_tree)
            else:
                lm_dist_mask = None

            matches_lm = knn_match_and_lowe_ratio_filter(Frame.matcher, des_i_lm, fr_j.des,
                                                         Frame.config_dict['lowe_ratio_test_threshold'],
                                                         dist_mask_12 = lm_dist_mask)


            dbg_str += "\t{} / {} existing landmarks".format(len(matches_lm),len(des_i_lm))
            l_i = []
            l_j = []
            for m in matches_lm:
                l_i += [m.queryIdx]
                l_j += [m.trainIdx]

            l_i_mask = np.zeros(len(fr_i.kp_lm_ind), dtype=bool)
            l_i_mask[l_i] = 1
            fr_i.kp_lm_ind_non_match = fr_i.kp_lm_ind[~l_i_mask]

            fr_i.kp_lm_ind = fr_i.kp_lm_ind[l_i_mask]  # l_i_1 is index of matched lm keypoints into fr_i.kp
                                                       # this has to be done this way since des_i_lm is a subset of fr1.des
            fr_i.lm_ind_non_match = fr_i.lm_ind[~l_i_mask]
            fr_i.lm_ind = fr_i.lm_ind[l_i_mask]
            fr_j.kp_m_prev_lm_ind = np.array(l_j)      # fr_j.des is the full list, so m.train gives us index into fr_j.kp

        Frame.frlog.info(dbg_str)
        Frame.frlog.info("{} landmarks couldn't be matched with matcher".format(len(fr_i.kp_lm_ind_non_match)))

            #Frame.frlog.debug("No of fr_i.kp_lm_ind: {}, fr_i.lm_ind: {}, fr_j.kp_m_prev_lm_ind: {}".format(
            #                  len(fr_i.kp_lm_ind),len(fr_i.lm_ind),len(fr_j.kp_m_prev_lm_ind)))

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
        None
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
        essmat_time = time.time()
        E, mask_e_all = cv2.findEssentialMat(kp_i_all_ud, kp_j_all_ud,
                                             focal=1.0, pp=(0., 0.),
                                             method=cv2.RANSAC, **Frame.config_dict['findEssential_settings'])
        Frame.frlog.info("Time to perform essential mat filter: {:.4f}".format(time.time()-essmat_time))

        essen_mat_pts = np.sum(mask_e_all)

        dbg_str = "Total -> Ess matrix : {} -> {}".format(len(kp_j_all_ud),essen_mat_pts)

        # getting the second mask for filtering using recover pose

        if Frame.config_dict['use_RecoverPose_Filter']:
            # Recover Pose filtering is breaking under certain conditions. Leave out for now.
            _, _, _, mask_RP_all = cv2.recoverPose(E, kp_i_all_ud, kp_j_all_ud, mask=mask_e_all)
            dbg_str += "\t Rec pose: {} -> {}".format(essen_mat_pts,np.sum(mask_RP_all))
        else:
            mask_RP_all = mask_e_all

        Frame.frlog.info(dbg_str)
        # Split the combined mask to lm feature mask and candidate mask
        mask_RP_lm = mask_RP_all[:num_landmarks]
        mask_RP_cand = mask_RP_all[-num_cand:]

        # Assign indexes
        fr_i.kp_lm_ind = fr_i.kp_lm_ind[mask_RP_lm[:,0].astype(bool)]
        fr_j.kp_m_prev_lm_ind = fr_j.kp_m_prev_lm_ind[mask_RP_lm[:,0].astype(bool)]

        fr_i.kp_cand_ind = fr_i.kp_cand_ind[mask_RP_cand[:,0].astype(bool)]
        fr_j.kp_m_prev_cand_ind = fr_j.kp_m_prev_cand_ind[mask_RP_cand[:,0].astype(bool)]

        fr_i.lm_ind = fr_i.lm_ind[mask_RP_lm[:,0].astype(bool)]

        Frame.frlog.info("Ess mat and RP filt: {} / {} landmarks and {} / {} candidates".format(np.sum(mask_RP_lm),
                                        num_landmarks, np.sum(mask_RP_cand), num_cand))

    @staticmethod
    def save_frame(fr_obj, file_name):
        '''
        This function saves Frame to the given file_name using the pickling function.

        Parameters
        ----------
        fr_obj: Frame
        file_name: string

        Returns
        -------
        None
        '''
        with open(file_name, 'wb') as output:
            pickle.dump(fr_obj, output)

    @staticmethod
    def load_frame(file_name):
        '''
        This function load a pre-processed Frame from a given file_name and returns it.

        Parameters
        ----------
        file_name: string

        Returns
        -------
        Frame
        '''
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
        fr1: Frame
        fr2: Frame

        Returns
        -------
        None
        '''
        # a. Computing Essential matrix from matched keypoints (kp) fr1 to fr2
        Frame.frlog.debug('Length of kp1: {}  Length of kp2: {}'.format(len(fr1.kp),len(fr2.kp)))

        Frame.match_and_propagate_keypoints(fr1,fr2,initialization=True)

        E_12, mask_e_12 = cv2.findEssentialMat(fr1.kp_ud_norm[fr1.kp_cand_ind],
                                               fr2.kp_ud_norm[fr2.kp_m_prev_cand_ind],
                                               focal=1.0, pp=(0., 0.), method=cv2.RANSAC,
                                               **Frame.config_dict['findEssential_settings'])

        Frame.frlog.info("Essential matrix: used {} of total {} matches".format(np.sum(mask_e_12),len(fr2.kp_m_prev_cand_ind)))
        essen_mat_pts = np.sum(mask_e_12)

        points, rot_2R1, trans_2t1, mask_RP_12 = cv2.recoverPose(E_12,
                                                                 fr1.kp_ud_norm[fr1.kp_cand_ind],
                                                                 fr2.kp_ud_norm[fr2.kp_m_prev_cand_ind],
                                                                 mask=mask_e_12)

        # b. Computing pose from Essential matrix between fr1 and fr2
        # c. Set the scale for algorithm as unit length between fr1 and fr2
        Frame.frlog.info("Recover pose used {} of total matches in Essential matrix {}".format(np.sum(mask_RP_12),essen_mat_pts))

        if fr1.T_groundtruth is not None:
            fr1.T_pnp = fr1.T_groundtruth
            Frame.scale = np.linalg.norm(fr2.T_groundtruth[:3,-1] - fr1.T_groundtruth[:3,-1])
            Frame.frlog.info("Since ground truth is available, setting scale to {:.2f} and first pose to: {}".format(Frame.scale, fr1.T_pnp))

        else:
            fr1.T_pnp = np.eye(4)
            Frame.scale = 1.0

        print(np.linalg.norm(trans_2t1))
        trans_2t1_scaled = trans_2t1 * Frame.scale

        pose_wT1 = fr1.T_pnp
        pose_2T1 = compose_T(rot_2R1,trans_2t1_scaled)
        pose_1T2 = T_inv(pose_2T1)
        pose_wT2 = pose_wT1 @ pose_1T2
        fr2.T_pnp = pose_wT2

        Frame.frlog.info("wT1 :\t"+str(fr1.T_pnp[:3,:]).replace('\n','\n\t\t'))
        Frame.frlog.info("wT2 :\t"+str(fr2.T_pnp[:3,:]).replace('\n','\n\t\t'))

        img12 = draw_point_tracks(fr1.kp[fr1.kp_cand_ind], fr2.gr,
                                  fr2.kp[fr2.kp_m_prev_cand_ind],
                                  mask_RP_12[:,0].astype(bool), False, color=[255,255,0])

        Frame.fig_frame_image = Frame.ax1.imshow(img12)

        plot_pose3_on_axes(Frame.ax2, pose_wT1, axis_length=0.5 * Frame.scale)

        Frame.cam_trail_pts = pose_wT2[:3,[-1]].T
        Frame.cam_pose_trail = plot_3d_points(Frame.ax2, Frame.cam_trail_pts, linestyle="", color='g', marker=".", markersize=2)

        if fr2.T_groundtruth is not None:
            Frame.cam_trail_gt_pts = fr2.T_groundtruth[:3,[-1]].T
        Frame.cam_pose_trail_gt = plot_3d_points(Frame.ax2, Frame.cam_trail_pts, linestyle="", color='orange', marker=".", markersize=2)

        #input("Press [enter] to continue.\n")

        fr1.kp_cand_ind         = fr1.kp_cand_ind[mask_RP_12[:,0].astype(bool)]
        fr2.kp_m_prev_cand_ind  = fr2.kp_m_prev_cand_ind[mask_RP_12[:,0].astype(bool)]

        # d. Triangulate landmarks using matched keypoints
        landmarks_12, mask_tri_12 = Frame.triangulate(fr1.T_pnp, fr2.T_pnp,
                                                      fr1.kp_ud_norm[fr1.kp_cand_ind],
                                                      fr2.kp_ud_norm[fr2.kp_m_prev_cand_ind],
                                                      None)

        Frame.landmarks = landmarks_12

        Frame.frlog.info("Triangulation used {} of total matches {} matches".format(np.sum(mask_tri_12),len(mask_tri_12)))

        img12_rej = draw_point_tracks(fr1.kp[fr1.kp_cand_ind], img12,
                                        fr2.kp[fr2.kp_m_prev_cand_ind],
                                        (1-mask_tri_12)[:,0].astype(bool), True, color=[255,0,0])

        if Frame.config_dict['plot_landmarks']:
            Frame.all_lm_plot_handle = plot_3d_points(Frame.ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='darkgrey')
        Frame.unused_lm_plot_handle = plot_3d_points(Frame.ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='r')
        Frame.new_lm_plot_handle = plot_3d_points(Frame.ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='g')

        set_axes_equal(Frame.ax2)
        Frame.fig2.canvas.draw_idle(); #plt.pause(0.01)
        Frame.cam_pose = plot_pose3_on_axes(Frame.ax2, pose_wT2, axis_length=1.0 * Frame.scale, center_plot=False, zoom_to_fit=False)
        Frame.update_plot_limits()
        Frame.fig2.canvas.draw_idle(); #plt.pause(0.01)

        #input("Press [enter] to continue.\n")
        #Frame.new_new_lm_plot_handle.remove()

        fr1.kp_cand_ind         = fr1.kp_cand_ind[mask_tri_12[:,0].astype(bool)]
        fr2.kp_m_prev_cand_ind  = fr2.kp_m_prev_cand_ind[mask_tri_12[:,0].astype(bool)]

        ### Add new landmarks to landmark array
        lm_list = []
        for (i,coord) in enumerate(landmarks_12):
            lm_list.append(landmark(fr1.frame_id,fr2.frame_id,
                                    fr1.kp_ud_norm[fr1.kp_cand_ind][[i]],
                                    fr2.kp_ud_norm[fr2.kp_m_prev_cand_ind][[i]],
                                    coord[None]))
        Frame.landmark_array=np.array(lm_list)

        fr2.partition_kp_cand()
        Frame.frlog.debug("Length of candidate pts: {}".format(len(fr2.kp_cand_ind)))

        if Frame.config_dict.get('display_candidates'):
            img12_cand = draw_points(img12_rej, fr2.kp[fr2.kp_cand_ind], color=[255,255,0])
        else:
            img12_cand = img12_rej

        Frame.fig_frame_image.set_data(img12_cand)
        Frame.fig1.canvas.draw_idle(); plt.pause(0.01)

        if len(Frame.landmarks) != len(fr2.kp_m_prev_cand_ind)  or len(fr2.kp_m_prev_cand_ind) != len(fr1.kp_cand_ind):
            raise ValueError('Between Frame {} and {}: Length of of kp_m_prev doesnt match kp_m_next or landmarks',format(fr1.frameid,fr2.frameid))

        fr2.lm_new_ind = np.array(range(len(fr2.kp_m_prev_cand_ind)))
        fr2.lm_ind = fr2.lm_new_ind
        fr2.kp_lm_ind = fr2.kp_m_prev_cand_ind

        input("Press [enter] to continue.")

    @staticmethod
    def process_keyframe_PNP(fr_i, fr_j):
        '''
        Performs the following operations on frame i and frame j
        a. match and propagate keypoints:
         1.matches points with existing landmarks in frame i and frame j and
           stores their index for respective frames
         2.matches new candidate points to triangulate in frame i and frame j
           and stores their index for respective frames
        b. combine and filter: combine the above two sets of points and filters
           them with essential matrix and recover pose method
        c. Slice inlier keypoints, undistort and compute pose using PNP Ransac algorithm

        Parameters
        ----------
        fr_i: Frame object
        fr_j: Frame object
        i = previous frame, j = current frame

        Returns
        -------
        None

        '''
        time_start = time.time()
        Frame.match_and_propagate_keypoints(fr_i, fr_j)
        Frame.frlog.debug("Time elapsed in match and prop keypoints: {:.4f}".format(time.time()-time_start))

        time_start = time.time()
        Frame.combine_and_filter(fr_i, fr_j)
        Frame.frlog.debug("Time elapsed in combine and filter: {:.4f}".format(time.time()-time_start))


        time_start = time.time()
        # Display landmarks carried from previous frame
        img_track_lm = draw_point_tracks(fr_i.kp[fr_i.kp_lm_ind],
                                         fr_j.gr,
                                         fr_j.kp[fr_j.kp_m_prev_lm_ind],
                                         None, False)

        # Display candidates carried from previous frame
        fr_j.img_track_all = draw_point_tracks(fr_i.kp[fr_i.kp_cand_ind],
                                          img_track_lm,
                                          fr_j.kp[fr_j.kp_m_prev_cand_ind],
                                          None, False, color=[255,255,0])

        Frame.fig_frame_image.set_data(fr_j.img_track_all)

        Frame.frlog.debug("Time elapsed in drawing tracks: {:.4f}".format(time.time()-time_start))
        time_start = time.time()

        # Slice inlier keypoints from fr_j.kp_ud_norm and use them for PNP calculation
        success, fr_j.T_pnp, mask_pnp = T_from_PNP_norm(Frame.landmarks[fr_i.lm_ind],
                                                        fr_j.kp_ud_norm[fr_j.kp_m_prev_lm_ind],
                                                        **Frame.config_dict['solvePnPRansac_settings']) # repErr = ceil2MSD(1/fr_j.gr.shape[1])

        if not success:
            Frame.frlog.critical("PNP failed in frame {}. Exiting...".format(fr_j.frame_id))
            exit()

        Frame.frlog.info((Fore.GREEN+"PNP inliers: {} / {} : {:.1f} %"+Style.RESET_ALL).format(np.sum(mask_pnp),len(fr_i.lm_ind),
                                                                                               np.sum(mask_pnp)/len(fr_i.lm_ind)*100))
        Frame.frlog.info("PNP_Pose:\t"+str(fr_j.T_pnp[:3,:]).replace('\n','\n\t\t\t'))

        plot_pose3_on_axes(Frame.ax2, fr_j.T_pnp, axis_length=1.0 * Frame.scale, center_plot=True, line_obj_list=Frame.cam_pose)

        Frame.cam_trail_pts = np.append(Frame.cam_trail_pts,fr_j.T_pnp[:3,[-1]].T,axis=0)
        plot_3d_points(Frame.ax2,Frame.cam_trail_pts , line_obj=Frame.cam_pose_trail, linestyle="", color='g', marker=".", markersize=2)
        Frame.update_plot_limits(fr_j.T_pnp[:3,[-1]].T)

        if fr_j.T_groundtruth is not None:
            Frame.cam_trail_gt_pts = np.append(Frame.cam_trail_gt_pts,fr_j.T_groundtruth[:3,[-1]].T,axis=0)
            plot_3d_points(Frame.ax2,Frame.cam_trail_gt_pts , line_obj=Frame.cam_pose_trail_gt)
            gt_trans_error = np.linalg.norm(fr_j.T_groundtruth[:3,-1]-fr_j.T_pnp[:3,-1])
            gt_rot_error = rotation_distance(fr_j.T_groundtruth[:3,:3], fr_j.T_pnp[:3,:3])
            Frame.frlog.info("Ground truth error: Trans: {:.5f} rot angle: {:.4f} deg".format(gt_trans_error,gt_rot_error))


        plot_3d_points(Frame.ax2, Frame.landmarks[fr_i.lm_ind[(mask_pnp[:,0]^1).astype(bool)]], line_obj=Frame.unused_lm_plot_handle)

        fr_i.lm_ind = fr_i.lm_ind[mask_pnp[:,0].astype(bool)]
        fr_i.kp_lm_ind = fr_i.kp_lm_ind[mask_pnp[:,0].astype(bool)]
        fr_j.kp_m_prev_lm_ind = fr_j.kp_m_prev_lm_ind[mask_pnp[:,0].astype(bool)]

        Frame.frlog.debug("Time elapsed in PNP: {:.4f}".format(time.time()-time_start))

        for (row,l) in enumerate(fr_i.lm_ind):
            Frame.landmark_array[l].add_observation(fr_j.frame_id, fr_j.kp_ud_norm[fr_j.kp_m_prev_lm_ind][[row]])

        fr_j.lm_ind = fr_i.lm_ind

    def process_keyframe_triangulation(fr_i, fr_j):
        '''
        Performs the following operations on frame i and frame j:
        a. Slice filtered candidate keypoints using index in respective frames
           and Triangulate them into new landmarks

        Parameters
        ----------
        fr_i: Frame object
        fr_j: Frame object
        i = previous frame, j = current frame

        Returns
        -------
        None

        '''

        time_start = time.time()

        # Slice and Triangulate
        lm_j_new, mask_tri = Frame.triangulate(fr_i.T_pnp, fr_j.T_pnp,
                                               fr_i.kp_ud_norm[fr_i.kp_cand_ind],
                                               fr_j.kp_ud_norm[fr_j.kp_m_prev_cand_ind],
                                               None)
        Frame.frlog.debug("Time elapsed in triangulate: {:.4f}".format(time.time()-time_start))
        time_start = time.time()

        Frame.frlog.info("Points triangulated: {} / {}".format(np.sum(mask_tri),len(mask_tri)))

        #if len(kp_prev_cand)>0:
        img_rej_pts = draw_point_tracks(fr_i.kp[fr_i.kp_cand_ind],
                                        fr_j.img_track_all,
                                        fr_j.kp[fr_j.kp_m_prev_cand_ind],
                                        (1-mask_tri)[:,0].astype(bool), True, color=[255,0,0])
        #else: img_rej_pts = img_track_all
        Frame.fig_frame_image.set_data(img_rej_pts)

        Frame.frlog.debug("Time elapsed in draw pt tracks: {:.4f} ".format(time.time()-time_start))
        time_start = time.time()

        fr_i.kp_cand_ind = fr_i.kp_cand_ind[mask_tri[:,0].astype(bool)]
        fr_j.kp_m_prev_cand_ind = fr_j.kp_m_prev_cand_ind[mask_tri[:,0].astype(bool)]


        plot_3d_points(Frame.ax2, lm_j_new, line_obj=Frame.new_lm_plot_handle)

        Frame.fig2.suptitle('Frame {}'.format(fr_j.frame_id))
        set_axes_equal(Frame.ax2, limits=Frame.ax2_plot_limits)

        Frame.fig2.canvas.draw_idle(); #plt.pause(0.01)

        num_curr_landmarks = len(Frame.landmarks)
        num_new_landmarks = len(lm_j_new)
        fr_j.lm_new_ind = np.array(range(num_curr_landmarks,num_curr_landmarks+num_new_landmarks))
        fr_i.kp_lm_ind = np.concatenate((fr_i.kp_lm_ind, fr_i.kp_cand_ind))
        fr_j.kp_lm_ind = np.concatenate((fr_j.kp_m_prev_lm_ind, fr_j.kp_m_prev_cand_ind))

        fr_j.lm_ind = np.concatenate((fr_j.lm_ind, fr_j.lm_new_ind))
        #print('previous indexes:',fr_j.kp_m_prev_lm_ind)
        Frame.landmarks = np.vstack((Frame.landmarks, lm_j_new))

        lm_new_list = []
        for (i,coord) in enumerate(lm_j_new):
            lm_new_list.append(landmark(fr_i.frame_id,fr_j.frame_id,
                                        fr_i.kp_ud_norm[fr_i.kp_cand_ind][[i]],
                                        fr_j.kp_ud_norm[fr_j.kp_m_prev_cand_ind][[i]],
                                        coord[None]))
        Frame.landmark_array=np.concatenate((Frame.landmark_array,np.array(lm_new_list)))

        # partition kp_m into two sets
        fr_j.partition_kp_cand()
        if Frame.config_dict.get('display_candidates'):
            img_cand_pts = draw_points(img_rej_pts,fr_j.kp[fr_j.kp_cand_ind],
                                       color=[255,255,0])
        else:
            img_cand_pts = img_rej_pts

        Frame.img_cand_pts = img_cand_pts
        Frame.fig_frame_image.set_data(img_cand_pts)
        Frame.ax1.set_title('Frame {}'.format(fr_j.frame_id))
        Frame.fig1.canvas.draw_idle(); #plt.pause(0.01)

        Frame.fig1.canvas.start_event_loop(0.001)
        Frame.fig2.canvas.start_event_loop(0.001)
