"""
GTSAM Copyright 2010-2018, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Simple robotics example using odometry measurements and bearing-range (laser) measurements
Author: Alex Cunningham (C++), Kevin Deng & Frank Dellaert (Python)
"""
# pylint: disable=invalid-name, E1101

#from __future__ import print_function

import math

import numpy as np

import gtsam

import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
from subprocess import check_call, call
import cv2


def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=np.float)

def plot_graph_poses(graph, result, axes):
    axes.cla()
    axes.set(adjustable='box', aspect='equal')
    axes.set_xlim(-1,5)
    axes.set_ylim(-1,3)
    marginals = gtsam.Marginals(graph, result)
    num_poses = result.keys().size()

    for i in range(1, num_poses+1):
        gtsam_plot.plot_pose2_on_axes(axes, result.atPose2(i), 0.5, marginals.marginalCovariance(i))
    plt.pause(3.)

graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

iS2params = gtsam.ISAM2Params()
iS2params.setRelinearizeThreshold(0.1)
iS2params.setRelinearizeSkip(1)
isam2 = gtsam.ISAM2(iS2params)

# Create noise models
PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(vector3(0.3, 0.3, 0.1))
ODOMETRY_NOISE = gtsam.noiseModel_Diagonal.Sigmas(vector3(0.2, 0.2, 0.1))

between_pose_list = [ gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), PRIOR_NOISE),
                      gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2, 0, 0), ODOMETRY_NOISE),
                      gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE),
                      gtsam.BetweenFactorPose2(3, 4, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE),
                      gtsam.BetweenFactorPose2(4, 5, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE),
                      gtsam.BetweenFactorPose2(5, 2, gtsam.Pose2(2, 0, math.pi / 2), ODOMETRY_NOISE)]

estimate_list = [ gtsam.Pose2(0.5, 0.0, 0.2),
                  gtsam.Pose2(2.3, 0.1, -0.2),
                  gtsam.Pose2(3.8, -0.25, math.pi / 2),
                  gtsam.Pose2(4.12, 2.22, math.pi),
                  gtsam.Pose2(2.2, -1.8, -math.pi / 2),
                  None] # Loop closure no new pose
plt.close(0)
fig, axes = plt.subplots(1,2,num=0)
axes[1].set_axis_off()


for i,(pose_factor, estimate) in enumerate(zip(between_pose_list,estimate_list),1):
    graph.add(pose_factor)
    if estimate is not None:
        initial_estimate.insert(i, estimate)

    if i == 1:
        plot_graph_poses(graph,initial_estimate, axes[0])
    isam2.update(graph, initial_estimate)
    current_estimate = isam2.calculateBestEstimate()
    graph.resize(0)
    initial_estimate.clear()

    isam2.saveGraph('isam_bayes_tree.dot')
    call(['/Users/vik748/miniconda3/envs/simple_vslam_env/bin/dot','-Tpng','isam_bayes_tree.dot','-o','isam_bayes_tree.png'])
    bt_img = cv2.imread('isam_bayes_tree.png')
    axes[1].imshow(bt_img)
    plot_graph_poses(isam2.getFactorsUnsafe(),current_estimate,axes[0])
    fig.savefig("plot_{}.png".format(i))


print("\nFactor Graph:\n{}".format(graph))  # print

print("\nInitial Estimate:\n{}".format(initial_estimate))  # print


# 5. Calculate and print marginal covariances for all variables
