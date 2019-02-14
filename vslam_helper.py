#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 08:36:51 2019
Simple VSLAM Helper File
@author: vik748
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import spatial


def compose_T(R,t):
    return np.vstack((np.hstack((R,t)),np.array([0, 0, 0, 1])))

def decompose_T(T_in):
    return T_in[:3,:3], T_in[:3,[-1]].T

def pose_inv(R_in, t_in):
    t_out = -np.matmul((R_in).T,t_in)
    R_out = R_in.T
    return R_out,t_out

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))

def triangulate(T_w_1, T_w_2, pts_1, pts_2 ):
    '''
    This function accepts two homogeneous transforms (poses) of 2 cameras in world coordinates,
    along with corresponding matching points and returns the 3D coordinates in world coordinates
    '''
    T_origin = np.eye(4)
    P_origin = T_origin[:3]
    # calculate the transform of 1 in 2's frame
    T_2_w = T_inv(T_w_2)
    T_2_1 = T_2_w @ T_w_1
    P_2_1 = T_2_1[:3]
    print ("P_2_1: ", P_2_1)
    
    # Calculate points in 0,0,0 frame
    pts_3d_frame1_hom = cv2.triangulatePoints(P_origin, P_2_1, pts_1, pts_2).T
    pts_3d_frame1_hom_norm = pts_3d_frame1_hom /  pts_3d_frame1_hom[:,-1][:,None]
    # Move 3d points to world frame by transforming with T_w_1
    pts_3d_w_hom = pts_3d_frame1_hom_norm @ T_w_1.T
    return pts_3d_w_hom[:, :3]

def T_from_PNP(coord_3d, img_pts, K, D):
    success, rvec_to_obj, tvecs_to_obj, inliers = cv2.solvePnPRansac(coord_3d, img_pts, K, D)

    if success:    
        R_to_obj, _ = cv2.Rodrigues(rvec_to_obj)
        return success, compose_T(*pose_inv(R_to_obj, tvecs_to_obj)), inliers
    else: 
        return success, None, None

def undistortKeyPoints(kps, K, D):
    '''
    This function extracts coordinates from keypoint object,
    undistorts them using K and D and returns undistorted coordinates"
    '''
    #kp_pts = np.array([o.pt for o in kps])
    #kp_pts_cont = np.ascontiguousarray(kp_pts[:,:2]).reshape((kp_pts.shape[0],1,2))
    # this version returns normalized points with F=1 and centered at 0,0
    # cv2.undistortPoints(kp_pts_cont, K, D,  noArray(), K) would return unnormalized output
    return	cv2.undistortPoints(np.expand_dims(kps, axis=1), 
                               cameraMatrix=K, distCoeffs=D)[:,0,:]

def displayMatches(img_left,kp1,img_right,kp2, matches, mask, display_invalid, in_image=None, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    bool_mask = mask.astype(bool)
    if in_image is None: mode_flag=0
    else: mode_flag =1
    img_valid = cv2.drawMatches(img_left,kp1,img_right,kp2,matches, in_image, 
                                matchColor=color, 
                                matchesMask=bool_mask.ravel().tolist(), flags=mode_flag)
    
    if display_invalid:
        img_valid = cv2.drawMatches(img_left,kp1,img_right,kp2,matches, img_valid, 
                                  matchColor=(255, 0, 0), 
                                  matchesMask=np.invert(bool_mask).ravel().tolist(), 
                                  flags=1)
    return img_valid

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def plot_pose3_on_axes(axes, T, axis_length=0.1):
    """Plot a 3D pose 4x4 homogenous transform  on given axis 'axes' with given 'axis_length'."""
    plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length)

def plot_pose3RT_on_axes(axes, gRp, origin, axis_length=0.1):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin, x_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin, y_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin, z_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')

def plot_3d_points(axes, vals, *args, **kwargs):
    graph, = axes.plot(vals[:,0], vals[:,1], vals[:,2], *args, **kwargs)
    return graph

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,2) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1],
                ...,
                [xn,yn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keept or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter


def tiled_features(kp, img, tiley, tilex):
    feat_per_cell = int(len(kp)/(tilex*tiley))
    HEIGHT, WIDTH = img.shape
    assert WIDTH%tiley == 0, "Width is not a multiple of tilex"
    assert HEIGHT%tilex == 0, "Height is not a multiple of tiley"
    w_width = int(WIDTH/tiley)
    w_height = int(HEIGHT/tilex)
        
    xx = np.linspace(0,HEIGHT-w_height,tilex,dtype='int')
    yy = np.linspace(0,WIDTH-w_width,tiley,dtype='int')
        
    kps = np.array([])
    pts = np.array([keypoint.pt for keypoint in kp])
    kp = np.array(kp)
    
    for ix in xx:
        for iy in yy:
            inbox_mask = bounding_box(pts, ix,ix+w_height,iy,iy+w_height)
            inbox = kp[inbox_mask]
            inbox_sorted = sorted(inbox, key = lambda x:x.response, reverse = True)
            inbox_sorted_out = inbox_sorted[:feat_per_cell]
            kps = np.append(kps,inbox_sorted_out)
    return kps.tolist()


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 15, color)
    return vis

def knn_match_and_filter(matcher, kp1, kp2, des1, des2):
    matches_knn = matcher.knnMatch(des1,des2, k=2)
    matches = []
    kp1_match = []
    kp2_match = []
    
    for i,match in enumerate(matches_knn):
        if len(match)>1:
            if match[0].distance < 0.80*match[1].distance:
                matches.append(match[0])
                kp1_match.append(kp1[match[0].queryIdx].pt)
                kp2_match.append(kp2[match[0].trainIdx].pt)
        elif len(match)==1:
            matches.append(match[0])
            kp1_match.append(kp1[match[0].queryIdx].pt)
            kp2_match.append(kp2[match[0].trainIdx].pt)

    return np.ascontiguousarray(kp1_match), np.ascontiguousarray(kp2_match), matches

def move_figure(position="top-right"):
    '''
    Move and resize a window to a set of standard positions on the screen.
    Possible positions are:
    top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    '''

    mgr = plt.get_current_fig_manager()
    ##mgr.full_screen_toggle()  # primitive but works to get screen size
    #mgr.window.showMaximized()
    #py = mgr.canvas.height()
    #px = mgr.canvas.width()
    py = 900
    px = 1600

    d = 10  # width of the window border in pixels
    if position == "top":
        # x-top-left-corner, y-top-left-corner, x-width, y-width (in pixels)
        mgr.window.setGeometry(d, 4*d, px - 2*d, py/2 - 4*d)
    elif position == "bottom":
        mgr.window.setGeometry(d, py/2 + 5*d, px - 2*d, py/2 - 4*d)
    elif position == "left":
        mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py - 4*d)
    elif position == "right":
        mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py - 4*d)
    elif position == "top-left":
        mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py/2 - 4*d)
    elif position == "top-right":
        mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py/2 - 4*d)
    elif position == "bottom-left":
        mgr.window.setGeometry(d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)
    elif position == "bottom-right":
        mgr.window.setGeometry(px/2 + d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)

def radial_non_max(kp_list, dist):
    kp = np.array(kp_list)
    kp_mask = np.ones(len(kp), dtype=bool)
    pts = [k.pt for k in kp]
    tree = spatial.cKDTree(pts)
    #print ("len of kp1:",len(kp))
    for i, k in enumerate(kp):
        if kp_mask[i]:
            pt = tree.data[i]
            idx = tree.query_ball_point(tree.data[i], dist, p=2., eps=0, n_jobs=1)
            resp = [kp[ii].response for ii in idx]
            _, maxi = max([(v,i) for i,v in enumerate(resp)])
            del idx[maxi]
            for kp_i in idx:
                kp_mask[kp_i] = False 
    return kp[kp_mask].tolist()