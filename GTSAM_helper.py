"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

A structure-from-motion problem on a simulated dataset
"""
from __future__ import print_function

import gtsam
import numpy as np
import SFMdata
from gtsam.gtsam import (Cal3_S2, Cal3DS2, DoglegOptimizer,
                         GenericProjectionFactorCal3_S2, NonlinearFactorGraph,
                         Point3, Pose3, PriorFactorPoint3, PriorFactorPose3,
                         Rot3, SimpleCamera, Values, symbolChr, symbolIndex, RangeFactorPose3)
from gtsam.utils import plot
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611
from matplotlib import pyplot as plt

class GraphOptimization():
    def __init__(self):
        self.graph = NonlinearFactorGraph()

        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z

        pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        x0factor = PriorFactorPose3(symbol('x', 0), poses[0], pose_noise)
        self.graph.push_back(x0factor)
        
        x0x1_noise = gtsam.noiseModel_Isotropic.Sigma(1, 0.1)
        x1factor = RangeFactorPose3(symbol('x', 0),symbol('x', 1), 15.0,x0x1_noise)
        graph.push_back(x1factor)


        self.opt_params = gtsam.DoglegParams()
        self.opt_params.setVerbosity('Error')
        self.opt_params.setErrorTol(0.1)
       

    def optimize(self, max_iterations=20):
        self.optimizer = DoglegOptimizer(self.graph, self.initial_estimate, self.opt_params)
        print('Optimizing:')
        self.result = self/optimizer.optimizeSafely()

    def add_vertex(self, id, pose, fixed=False):
        v_se2 = g2o.VertexSE2()
        v_se2.set_id(id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(fixed)
        super().add_vertex(v_se2)

    def add_edge(self, vertices, measurement, 
            information=np.identity(3),
            robust_kernel=None):

        edge = g2o.EdgeSE2()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()

def pt_dist(pt1, pt2):
    return ( (pt1.x()-pt2.x())*(pt1.x()-pt2.x()) + (pt1.y()-pt2.y())*(pt1.y()-pt2.y()) + (pt1.z()-pt2.z())*(pt1.z()-pt2.z()) )**.5

def symbol(name: str, index: int) -> int:
    """ helper for creating a symbol without explicitly casting 'name' from str to int """
    return gtsam.symbol(ord(name), index)

def key_label(key_id):
    return str(chr(symbolChr(key_id))), symbolIndex(key_id)

def plot_line_on_axes(axes, point1, point2, linespec):
    """Plot a 3D point on given axis 'axes' with given 'linespec'."""
    xs = np.array([point1.x(), point2.x()])
    ys = np.array([point1.y(), point2.y()])
    zs = np.array([point1.z(), point2.z()])
    
    axes.plot(xs, ys, zs, linespec)

def plot_line(fignum, point1, point2, linespec):
    """Plot a 3D point on given figure with given 'linespec'."""
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plot_line_on_axes(axes, point1, point2, linespec)

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

def Point2arr(pt):
    if type(pt) is gtsam.gtsam.Point2 :
        return np.array([pt.x(), pt.y()])
    elif type(pt) is gtsam.gtsam.Point3 :
        return np.array([pt.x(), pt.y(), pt.z()])
    else:
        raise ValueError("Point supplied is neither gtsam.Point2 or gtsam.Point3")

def arr2Point(pt):
    if len(pt) == 2 :
        return gtsam.gtsam.Point2(pt)
    elif len(pt) ==3 :
        return gtsam.gtsam.Point3(pt)
    else:
        raise ValueError("Length of Array supplied is neither 2 or 3")

def main():
    """
    Camera observations of landmarks (i.e. pixel coordinates) will be stored as Point2 (x, y).

    Each variable in the system (poses and landmarks) must be identified with a unique key.
    We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
    Here we will use Symbols

    In GTSAM, measurement functions are represented as 'factors'. Several common factors
    have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
    Here we will use Projection factors to model the camera's landmark observations.
    Also, we will initialize the robot at some location using a Prior factor.

    When the factors are created, we will add them to a Factor Graph. As the factors we are using
    are nonlinear factors, we will need a Nonlinear Factor Graph.

    Finally, once all of the factors have been added to our factor graph, we will want to
    solve/optimize to graph to find the best (Maximum A Posteriori) set of variable values.
    GTSAM includes several nonlinear optimizers to perform this step. Here we will use a
    trust-region method known as Powell's Degleg

    The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
    nonlinear functions around an initial linearization point, then solve the linear system
    to update the linearization point. This happens repeatedly until the solver converges
    to a consistent set of variable values. This requires us to specify an initial guess
    for each variable, held in a Values container.
    """


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
        