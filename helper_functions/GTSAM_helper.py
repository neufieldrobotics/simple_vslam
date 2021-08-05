"""
Wrappers for GTSAM for seamless implementation with python/numpy
"""
from __future__ import print_function

import gtsam
import numpy as np
# import SFMdata
from gtsam.gtsam import (Cal3_S2, Cal3DS2, DoglegOptimizer,
                         GenericProjectionFactorCal3_S2, NonlinearFactorGraph,
                         Pose3, PriorFactorPoint3, PriorFactorPose3,
                         Rot3, Values, symbolChr, symbolIndex, RangeFactorPose3)
import gtsam.utils.plot as gtsam_plot
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611
from matplotlib import pyplot as plt
import cv2

from frame import Frame


class iSAM2Wrapper():
    """
    GTSAM wrapper class that aids seamless integration with
    python/numpy

    Contains helper functions that act as a layer of abstraction
    around GTSAM
    """

    def __init__(self, pose0=np.eye(4), pose0_to_pose1_range=1.0, K=np.eye(3),
                 relinearizeThreshold=0.1, relinearizeSkip=10, proj_noise_val=1.0):
        self.graph = NonlinearFactorGraph()

        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z

        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        x0factor = PriorFactorPose3(iSAM2Wrapper.get_key('x', 0), gtsam.gtsam.Pose3(pose0), pose_noise)
        self.graph.push_back(x0factor)

        # Set scale between pose 0 and pose 1 to Unity 
        x0x1_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
        x1factor = RangeFactorPose3(iSAM2Wrapper.get_key('x', 0), iSAM2Wrapper.get_key('x', 1), pose0_to_pose1_range,
                                    x0x1_noise)
        self.graph.push_back(x1factor)

        iS2params = gtsam.ISAM2Params()
        iS2params.setRelinearizeThreshold(relinearizeThreshold)
        iS2params.setRelinearizeSkip(relinearizeSkip)
        self.isam2 = gtsam.ISAM2(iS2params)

        self.projection_noise = gtsam.noiseModel.Isotropic.Sigma(2, proj_noise_val)
        self.K = gtsam.Cal3_S2(iSAM2Wrapper.CameraMatrix_to_Cal3_S2(K))
        # self.opt_params = gtsam.DoglegParams()
        # self.opt_params.setVerbosity('Error')
        # self.opt_params.setErrorTol(0.1)

        self.initial_estimate = gtsam.Values()
        self.initial_estimate.insert(iSAM2Wrapper.get_key('x', 0),
                                     gtsam.gtsam.Pose3(pose0))
        self.lm_factor_ids = set()

    def add_GenericProjectionFactorCal3_S2_factor(self, pt_uv, X_id, L_id):
        """
        Adds a landmark point to factor graph

        Function adds a set of Landmark points in succession. The
        factor is generated using GTSAM function
        GenericProjectionFactorCal3_S2. The factor is pushed back
        into self.graph

        :param pt_uv: ndarray, set of landmark points of shape Nx2
        :param X_id: int, key number of pose from which landmarks are
        observed
        :param L_id: int64 ndarray, array of landmark key numbers
        """

        if pt_uv.ndim == 1:
            raise ValueError("Supplied point is 1-dimensional, required Nx2 array")
        if pt_uv.shape[1] != 2:
            raise ValueError("2nd dimension on supplied point is not 2, required Nx2 array")
        for pt, l in zip(pt_uv, L_id):
            fact = gtsam.GenericProjectionFactorCal3_S2(pt,
                                                        self.projection_noise,
                                                        self.get_key('x', X_id),
                                                        self.get_key('l', l),
                                                        self.K)
            self.graph.push_back(fact)
        self.lm_factor_ids.update(L_id)

    def add_PoseEstimate(self, X_id, T):
        """
        Adds a pose estimate to self.initial_estimate

        :param X_id: int, key number of pose to be added
        :param T: float64 ndarray, 4x4 transformation matrix
        """

        self.initial_estimate.insert(iSAM2Wrapper.get_key('x', X_id),
                                     gtsam.gtsam.Pose3(T))

    def add_LandmarkEstimate(self, L_id, Pt_estimate):
        """
        Adds landmark estimates to self.initial_estimate

        :param L_id: int64 ndarray, array of landmark key numbers
        :param Pt_estimate: float64 ndarray, set of landmark points
        """

        if Pt_estimate.ndim == 1:
            raise ValueError("Supplied point is 1-dimensional, required Nx3 array")
        if Pt_estimate.shape[1] != 3:
            raise ValueError("2nd dimension on supplied point is not 3, required Nx3 array")
        for l, p_est in zip(L_id, Pt_estimate):
            self.initial_estimate.insert(iSAM2Wrapper.get_key('l', l),
                                         p_est)

    def update(self, iterations=1):
        """
        Performs ISAM2 update steps

        Number of update steps given by parameter iterations.

        :param iterations: int, default=1. number of update steps
        """

        self.isam2.update(self.graph, self.initial_estimate)
        # Perform additional iterations as specified
        for i in range(2, iterations + 1):
            self.isam2.update()
        # self.current_estimate = self.isam2.calculateEstimate()
        self.current_estimate = self.isam2.calculateBestEstimate()
        self.graph.resize(0)
        self.initial_estimate.clear()

    def get_Estimate(self):
        """
        Returns current estimate

        :return: self.current estimate
        """

        return self.current_estimate

    def get_landmark_estimates(self):
        """
        Function to get landmarks and their ids

        :return: [landmarks, landmark ids]
        """

        lm = []
        for l_id in self.lm_factor_ids:
            lm += [self.current_estimate.atPoint3(iSAM2Wrapper.get_key('l', l_id))]
        return np.array(lm), list(self.lm_factor_ids)

    def get_curr_Pose_Estimate(self, x_id):
        """
        Function to get current pose estimate

        :param x_id: int, key number of pose to be returned

        :return: current estimate of given pose id
        """

        return self.current_estimate.atPose3(iSAM2Wrapper.get_key('x', x_id)).matrix()

    def plot_estimate(self, fignum=0):
        """
        VisualISAMPlot plots current state of ISAM2 object
        Author: Ellon Paiva
        Based on MATLAB version by: Duy Nguyen Ta and Frank Dellaert
        """

        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()

        # Plot points
        # Can't use data because current frame might not see all points
        # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
        # gtsam.plot_3d_points(result, [], marginals)
        gtsam_plot.plot_3d_points(fignum, self.current_estimate, 'rx')

        # Plot cameras
        i = 0
        while self.current_estimate.exists(iSAM2Wrapper.get_key('x', i)):
            pose_i = self.current_estimate.atPose3(iSAM2Wrapper.get_key('x', i))
            gtsam_plot.plot_pose3(fignum, pose_i, 10)
            i += 1

        # draw
        # axes.set_xlim3d(-40, 40)
        # axes.set_ylim3d(-40, 40)
        # axes.set_zlim3d(-40, 40)
        axes.view_init(90, 0)

        plt.pause(.01)

    @staticmethod
    def pt_dist(pt1, pt2):
        """
        Returns the euclidean distance between two 3D points

        :param pt1: First point
        :param pt2: Second point

        :return: Euclidean distance between pt1 and pt2
        """

        return ((pt1.x() - pt2.x()) * (pt1.x() - pt2.x()) + (pt1.y() - pt2.y()) * (pt1.y() - pt2.y()) + (
                    pt1.z() - pt2.z()) * (pt1.z() - pt2.z())) ** .5

    @staticmethod
    def symbol(name: str, index: int) -> int:
        """ helper for creating a symbol without explicitly casting 'name' from str to int """

        return gtsam.symbol(ord(name), index)

    @staticmethod
    def key_label(key_id):
        """
        Separates symbol character and symbol index

        :param key_id: symbol to be separated

        :return: symbol character, symbol index
        """

        return str(chr(symbolChr(key_id))), symbolIndex(key_id)

    @staticmethod
    def plot_line_on_axes(axes, point1, point2, linespec):
        """Plot a 3D point on given axis 'axes' with given 'linespec'."""

        xs = np.array([point1.x(), point2.x()])
        ys = np.array([point1.y(), point2.y()])
        zs = np.array([point1.z(), point2.z()])

        axes.plot(xs, ys, zs, linespec)

    @staticmethod
    def plot_line(fignum, point1, point2, linespec):
        """Plot a 3D point on given figure with given 'linespec'."""

        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plot_line_on_axes(axes, point1, point2, linespec)

    @staticmethod
    def draw_graph(graph, values, fig_num=1):
        """
        plots the poses given in values
        """

        for i in range(graph.size()):
            factor = graph.at(i)
            fkeys = factor.keys()
            if fkeys.size() == 2:
                x = values.atPose3(fkeys.at(0)).translation()
                if key_label(fkeys.at(1))[0] is 'x':
                    l = values.atPose3(fkeys.at(1)).translation()
                else:
                    l = values.atPoint3(fkeys.at(1))
                plot_line(fig_num, x, l, 'b')
        poses = gtsam.allPose3s(values)
        for i in range(poses.size()):
            plot.plot_pose3(fig_num, poses.atPose3(poses.keys().at(i)), axis_length=2.0)

    @staticmethod
    def get_key(letter, number):
        """
        Create key for pose id number

        :param letter: letter of key to be retrieved
        :param number: number of key to be retrieved
        :return: key
        """

        return int(gtsam.symbol(letter, number))

    @staticmethod
    def CameraMatrix_to_Cal3_S2(K):
        """
        Convert 3x3 camera matrix to 5 length vector for Cal3_S2 given as
        Cal3_S2 (double fx, double fy, double s, double u0, double v0)

        :param K: 3x3 camera matrix
        :return: camera matrix in Cal3_S2 format
        """

        if not np.allclose(K, np.triu(K)):
            raise ValueError('K matrix not upper triangular, might be incorrrect')
        fx, fy, s, u0, v0 = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]
        return np.array([fx, fy, s, u0, v0])

    '''      
    def add_keyframe_factors(self, frame_queue, initialization=False):
        fr_j = frame_queue.queue[-1]
        fr_i = frame_queue.queue[-2]
        #fr_h = frame_queue[-3]
        
        if not initialization:
            ## Add exsisting landmarks        
            #  Add projection factors only to frame j
            pt_uv = cv2.undistortPoints(np.expand_dims(fr_j.kp[fr_j.kp_m_prev_lm_ind],1),
                                                       Frame.K, Frame.D)[:,0,:]
            self.add_GenericProjectionFactorCal3_S2_factor(pt_uv, fr_j.frame_id, fr_i.lm_ind)
        
        ## Add new landmarks to frame i and j
        #  Add projection factors to frame i
        print ("k:", Frame.K)
        pt_uv = cv2.undistortPoints(np.expand_dims(fr_i.kp[fr_i.kp_cand_ind],1),
                                                   Frame.K, Frame.D)[:,0,:]
        self.add_GenericProjectionFactorCal3_S2_factor(pt_uv, fr_i.frame_id, fr_j.lm_new_ind)

        #  Add projection factors to frame j
        pt_uv = cv2.undistortPoints(np.expand_dims(fr_j.kp[fr_j.kp_m_prev_cand_ind],1),
                                                   Frame.K, Frame.D)[:,0,:]
        self.add_GenericProjectionFactorCal3_S2_factor(pt_uv, fr_j.frame_id, fr_j.lm_new_ind)
        
        ## Add estiamates
        # Add landmark estimates for the newly created landmarks
        self.add_LandmarkEstimate(fr_j.lm_new_ind, Frame.landmarks[fr_j.lm_new_ind])
        # Add pose estimate from the new frame j
        self.add_PoseEstimate(fr_j.frame_id, fr_j.T_pnp)
    '''

    def add_keyframe_factors(self, fr_j):
        """
        Adds a landmark to factor graph and initial estimate if it's
        observed in at least three frames

        :param fr_j: current operational frame
        """

        new_lm_fact = 0
        exist_lm_fact = 0
        for l, l_id in zip(Frame.landmark_array[fr_j.lm_ind], fr_j.lm_ind):
            if len(l.observed_kps) == 3:
                ## Add new landmarks to GTSAM when they have been observed in 3 frames
                # Add from frame j to landmark
                self.add_GenericProjectionFactorCal3_S2_factor(l.keypoint_in_frame(fr_j.frame_id),
                                                               fr_j.frame_id,
                                                               [l_id])
                # Add from frame i to landmark
                self.add_GenericProjectionFactorCal3_S2_factor(l.keypoint_in_frame(fr_j.frame_id - 1),
                                                               fr_j.frame_id - 1,
                                                               [l_id])
                # Add from frame h to landmark
                self.add_GenericProjectionFactorCal3_S2_factor(l.keypoint_in_frame(fr_j.frame_id - 2),
                                                               fr_j.frame_id - 2,
                                                               [l_id])
                # Add landmark estimates for the newly created landmarks
                self.add_LandmarkEstimate([l_id], l.coord_3d)
                # Frame.update_plot_limits(l.coord_3d)
                new_lm_fact += 1

            elif len(l.observed_kps) > 3:
                #  Add projection factors only to frame j  for landmarks already added to GTSAM before
                self.add_GenericProjectionFactorCal3_S2_factor(l.keypoint_in_frame(fr_j.frame_id),
                                                               fr_j.frame_id,
                                                               [l_id])
                exist_lm_fact += 1
        ## Add estiamates
        self.add_PoseEstimate(fr_j.frame_id, fr_j.T_pnp)
        Frame.frlog.info("GTSAM add factors: existing(>3 obs): {}   new (3 obs): {}".format(exist_lm_fact, new_lm_fact))
