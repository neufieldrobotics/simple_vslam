"""
Wrappers for GTSAM cython for seamless implementation with python/numpy
"""
from __future__ import print_function

import gtsam
import numpy as np
#import SFMdata
from gtsam.gtsam import (Cal3_S2, Cal3DS2, DoglegOptimizer,
                         GenericProjectionFactorCal3_S2, NonlinearFactorGraph,
                         Point3, Pose3, PriorFactorPoint3, PriorFactorPose3,
                         Rot3, SimpleCamera, Values, symbolChr, symbolIndex, RangeFactorPose3)
import gtsam.utils.plot as gtsam_plot
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611
from matplotlib import pyplot as plt
import cv2
from helper_functions.frame import Frame

class iSAM2Wrapper():
    def __init__(self,pose0=np.eye(4),pose0_to_pose1_range = 1.0, K=np.eye(3),
                 relinearizeThreshold=0.1,relinearizeSkip=1, proj_noise_val=1.0):
        self.graph = NonlinearFactorGraph()

        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z

        pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        x0factor = PriorFactorPose3(iSAM2Wrapper.get_key('x', 0), gtsam.gtsam.Pose3(pose0), pose_noise)
        self.graph.push_back(x0factor)
        
        # Set scale between pose 0 and pose 1 to Unity 
        x0x1_noise = gtsam.noiseModel_Isotropic.Sigma(1, 0.1)
        x1factor = RangeFactorPose3(iSAM2Wrapper.get_key('x', 0),iSAM2Wrapper.get_key('x', 1), pose0_to_pose1_range,x0x1_noise)
        self.graph.push_back(x1factor)

        iS2params = gtsam.ISAM2Params()
        iS2params.setRelinearizeThreshold(relinearizeThreshold)
        iS2params.setRelinearizeSkip(relinearizeSkip)
        self.isam2 = gtsam.ISAM2(iS2params)
        
        self.projection_noise = gtsam.noiseModel_Isotropic.Sigma(2, proj_noise_val)
        self.K = gtsam.Cal3_S2(iSAM2Wrapper.CameraMatrix_to_Cal3_S2(K))
        #self.opt_params = gtsam.DoglegParams()
        #self.opt_params.setVerbosity('Error')
        #self.opt_params.setErrorTol(0.1)
       
        self.initial_estimate = gtsam.Values()
        self.initial_estimate.insert(iSAM2Wrapper.get_key('x',0), 
                                     gtsam.gtsam.Pose3(pose0))
        

    def add_GenericProjectionFactorCal3_S2_factor(self, pt_uv, X_id, L_id):
        if pt_uv.ndim == 1: 
            raise ValueError("Supplied point is 1-dimensional, required Nx2 array")
        if pt_uv.shape[1] != 2:
            raise ValueError("2nd dimension on supplied point is not 2, required Nx2 array")
        for pt,l in zip(pt_uv, L_id):
            fact = gtsam.GenericProjectionFactorCal3_S2(gtsam.gtsam.Point2(*pt), 
                                                        self.projection_noise, 
                                                        self.get_key('x', X_id), 
                                                        self.get_key('l', l), 
                                                        self.K)    
            self.graph.push_back(fact)
        
    def add_PoseEstimate(self, X_id, T):    
        self.initial_estimate.insert(iSAM2Wrapper.get_key('x',X_id), 
                                     gtsam.gtsam.Pose3(T))
    
    def add_LandmarkEstimate(self, L_id, Pt_estimate):
        if Pt_estimate.ndim == 1: 
            raise ValueError("Supplied point is 1-dimensional, required Nx3 array")
        if Pt_estimate.shape[1] != 3:
            raise ValueError("2nd dimension on supplied point is not 3, required Nx3 array")
        for l, p_est in zip(L_id, Pt_estimate):
            self.initial_estimate.insert(iSAM2Wrapper.get_key('l',l), 
                                         gtsam.Point3(*p_est))
        
    def update(self,iterations = 1):
        self.isam2.update(self.graph, self.initial_estimate)
        # Perform additional iterations as specified
        for i in range(2,iterations+1):
            self.isam2.update()
        self.current_estimate = self.isam2.calculateEstimate()
        self.graph.resize(0)
        self.initial_estimate.clear()

    def get_Estimate(self):    
        return self.current_estimate
    
    def get_curr_Pose_Estimate(self,x_id):    
        return self.current_estimate.atPose3(iSAM2Wrapper.get_key('x',x_id)).matrix()
    
    def plot_estimate(self,fignum = 0):
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
        while self.current_estimate.exists(iSAM2Wrapper.get_key('x',i)):
            pose_i = self.current_estimate.atPose3(iSAM2Wrapper.get_key('x',i))
            gtsam_plot.plot_pose3(fignum, pose_i, 10)
            i += 1
    
        # draw
        axes.set_xlim3d(-40, 40)
        axes.set_ylim3d(-40, 40)
        axes.set_zlim3d(-40, 40)
        axes.view_init(90, 0)
        plt.pause(.01)
    
    @staticmethod
    def pt_dist(pt1, pt2):
        return ( (pt1.x()-pt2.x())*(pt1.x()-pt2.x()) + (pt1.y()-pt2.y())*(pt1.y()-pt2.y()) + (pt1.z()-pt2.z())*(pt1.z()-pt2.z()) )**.5

    @staticmethod
    def symbol(name: str, index: int) -> int:
        """ helper for creating a symbol without explicitly casting 'name' from str to int """
        return gtsam.symbol(ord(name), index)
    
    @staticmethod
    def key_label(key_id):
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
        plot_line_on_axes(axes, point1, padd_keyframe_factorsoint2, linespec)

    @staticmethod
    def draw_graph(graph,values,fig_num=1):
        for i in range(graph.size()):
            factor = graph.at(i)
            fkeys = factor.keys()
            if fkeys.size() == 2:
                x = values.atPose3(fkeys.at(0)).translation()
                if key_label(fkeys.at(1))[0] is 'x':
                    l = values.atPose3(fkeys.at(1)).translation()
                else:
                    l = values.atPoint3(fkeys.at(1))
                plot_line(fig_num,x,l,'b')
        poses = gtsam.allPose3s(values)
        for i in range(poses.size()):
            plot.plot_pose3(fig_num, poses.atPose3(poses.keys().at(i)),axis_length=2.0)

    @staticmethod
    def Point2arr(pt):
        if type(pt) is gtsam.gtsam.Point2 :
            return np.array([pt.x(), pt.y()])
        elif type(pt) is gtsam.gtsam.Point3 :
            return np.array([pt.x(), pt.y(), pt.z()])
        else:
            raise ValueError("Point supplied is neither gtsam.Point2 or gtsam.Point3")
    @staticmethod
    def arr2Point(pt):
        if len(pt) == 2 :
            return gtsam.gtsam.Point2(pt)
        elif len(pt) ==3 :
            return gtsam.gtsam.Point3(pt)
        else:
            raise ValueError("Length of Array supplied is neither 2 or 3")
            
    @staticmethod
    def get_key(letter,number):
        """Create key for pose id number."""
        return int(gtsam.symbol(ord(letter), number))

    @staticmethod
    def CameraMatrix_to_Cal3_S2(K):
        '''
        Convert 3x3 camera matrix to 5 length vector for Cal3_S2 given as 
        Cal3_S2 (double fx, double fy, double s, double u0, double v0)
        '''
        if not np.allclose(K, np.triu(K)):
            raise ValueError('K matrix not upper triangular, might be incorrrect')
        fx,fy,s,u0,v0 = K[0,0],K[1,1],K[0,1],K[0,2],K[1,2]
        return np.array([fx,fy,s,u0,v0])
       
    def add_keyframe_factors(self, fr_i, fr_j, initialization=False):
        
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
if __name__ == '__main__':
    #main()
        # Define the camera calibration parameters
    K_sim = Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)
    K = Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
    K_cam = np.array([[50.0, 0.0, 50.0],
                     [0.0, 50.0, 50.0],
                     [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create the set of ground-truth landmarks
    points = SFMdata.createPoints()

    # Create the set of ground-truth poses
    poses = SFMdata.createPoses(K)

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    factor = PriorFactorPose3(symbol('x', 0), poses[0], pose_noise)
    graph.push_back(factor)

    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        camera = SimpleCamera(pose, K_sim)
        for j, point in enumerate(points):
            if pt_dist(point,pose.translation())<38.2:
                measurement = camera.project(point)
                undist_m = cv2.undistortPoints(np.expand_dims(np.expand_dims(Point_to_arr(measurement),0),1),K_cam,D)[0,0,:]
                undist_pt = arr2Point(undist_m)
                print("measurement: ",measurement)
                factor = GenericProjectionFactorCal3_S2(
                    undist_pt, measurement_noise, symbol('x', i), symbol('l', j), K)
                graph.push_back(factor)

    # Because the structure-from-motion problem has a scale ambiguity, the problem is still under-constrained
    # Here we add a prior on the position of the first landmark. This fixes the scale by indicating the distance
    # between the first camera and the first landmark. All other landmark positions are interpreted using this scale.
    #point_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)
    #factor = PriorFactorPoint3(symbol('l', 0), points[0], point_noise)
    #graph.push_back(factor)
    
    x0x1_noise = gtsam.noiseModel_Isotropic.Sigma(1, 0.1)
    factor = RangeFactorPose3(symbol('x', 0),symbol('x', 1), 15.0,x0x1_noise)
    graph.push_back(factor)

    #graph.print_('Factor Graph:\n')

    # Create the data structure to hold the initial estimate to the solution
    # Intentionally initialize the variables off from the ground truth
    initial_estimate = Values()
    for i, pose in enumerate(poses):
        r = Rot3.Rodrigues(np.random.random(3)) #r = Rot3.Rodrigues(-0.1, 0.2, 0.25)
        t = Point3(np.random.random(3)*1)#t = Point3(0.05, -0.10, 0.20)
        transformed_pose = pose.compose(Pose3(r, t))
        initial_estimate.insert(symbol('x', i), transformed_pose)
    for j, point in enumerate(points):
        transformed_point = Point3(point.vector() + np.random.random(3)*10) #np.array([-0.25, 0.20, 0.15]))
        initial_estimate.insert(symbol('l', j), transformed_point)
    initial_estimate.print_('Initial Estimates:\n')
    draw_graph(graph,initial_estimate,1)


    # Optimize the graph and print results
    params = gtsam.DoglegParams()
    params.setVerbosity('Error')
    params.setErrorTol(0.1)
    optimizer = DoglegOptimizer(graph, initial_estimate, params)
    print('Optimizing:')
    result = optimizer.optimizeSafely()
    #result.print_('Final results:\n')
    print('\n\ninitial error = {}'.format(graph.error(initial_estimate)))
#    print('\n\nfinal error = {}'.format(graph.error(result)))
    

    draw_graph(graph,result,2)
 '''      