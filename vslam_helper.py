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
import logging
import math

def R2d_from_theta(theta):  
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

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

'''
def triangulate(T_w_1, T_w_2, pts_1_2d, pts_2_2d, mask):
    '/''
    This function accepts two homogeneous transforms (poses) of 2 cameras in world coordinates,
    along with corresponding matching points and returns the 3D coordinates in world coordinates
    Mask must be a dimensionless array or n, array
    '/''
    vslog = logging.getLogger('VSLAM')
    pts_1 = np.expand_dims(pts_1_2d,1)
    pts_2 = np.expand_dims(pts_2_2d,1)
    T_origin = np.eye(4)
    P_origin = T_origin[:3]
    # calculate the transform of 1 in 2's frame
    T_2_w = T_inv(T_w_2)
    T_2_1 = T_2_w @ T_w_1
    P_2_1 = T_2_1[:3]
    vslog.info("P_2_1:\t"+str(P_2_1).replace('\n','\n\t\t\t'))
    
    t_2_1 = T_2_1[:3,-1]
    vslog.debug("No of points in pts_1: {}".format(pts_1.shape))
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
'''

def T_from_PNP(coord_3d, img_pts, K, D, **solvePnPRansac_settings):
    success, rvec_to_obj, tvecs_to_obj, inliers = cv2.solvePnPRansac(coord_3d, img_pts, 
                                                   K, D, **solvePnPRansac_settings)
    if success:    
        R_to_obj, _ = cv2.Rodrigues(rvec_to_obj)
        mask = np.zeros([len(img_pts),1],dtype='uint8')
        #mask[inliers[:,0]]=True
        mask[inliers[:,0],0]=1

        return success, compose_T(*pose_inv(R_to_obj, tvecs_to_obj)), mask#, coord_3d[inliers], img_pts[inliers]
    else: 
        return success, None, None

def T_from_PNP_norm(coord_3d, img_pts, **solvePnPRansac_settings):
    success, rvec_to_obj, tvecs_to_obj, inliers = cv2.solvePnPRansac(coord_3d, img_pts, 
                                                   np.eye(3), None, **solvePnPRansac_settings)
    if success:    
        R_to_obj, _ = cv2.Rodrigues(rvec_to_obj)
        mask = np.zeros([len(img_pts),1],dtype='uint8')
        mask[inliers[:,0],0]=1

        return success, compose_T(*pose_inv(R_to_obj, tvecs_to_obj)), mask#, coord_3d[inliers], img_pts[inliers]
    else: 
        return success, None, None

def ceil2MSD(x):
    mlp = 10**math.floor(math.log10(x))
    return float("%.0e" % (math.ceil(x / mlp ) * mlp)) 

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

def draw_points(vis_orig, points, color = (0, 255, 0), thick = 3):
    if vis_orig.shape[2] == 3: vis = vis_orig
    else: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    rad = int(vis.shape[1]/200)
    thick = round(vis.shape[1]/1000)

    for pt in points:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), rad , color, thickness=thick)
    return vis

def draw_arrows(vis_orig, points1, points2, color = (0, 255, 0), thick = 3):
    if len(vis_orig.shape) == 2: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    else: vis = vis_orig
    thick = round(vis.shape[1]/1000)+1
    for p1,p2 in zip(points1,points2):
        cv2.arrowedLine(vis, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color=color, thickness=thick)
    return vis

def draw_feature_tracks(img_left,kp1,img_right,kp2, matches, mask, display_invalid=False, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    bool_mask = mask.astype(bool)
    valid_right_matches = np.array([kp2[mat.trainIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    valid_left_matches = np.array([kp1[mat.queryIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    #img_right_out = draw_points(img_right, valid_right_matches)
    img_right_out = draw_arrows(img_right, valid_left_matches, valid_right_matches)
    
    
    return img_right_out

def draw_point_tracks(kp1,img_right,kp2, bool_mask=None, display_invalid=False, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    #bool_mask = mask[:,0].astype(bool)
    if bool_mask is None:
        valid_left_matches = kp1
        valid_right_matches = kp2
    else:
        valid_left_matches = kp1[bool_mask,:]
        valid_right_matches = kp2[bool_mask,:]
    #img_right_out = draw_points(img_right, valid_right_matches)
    img_right_out = draw_arrows(img_right, valid_left_matches, valid_right_matches, color=color)
    
    return img_right_out

def center_3d_plot_around_pt(ax, origin):
    xmin,xmax = ax.get_xlim3d()
    ymin,ymax = ax.get_ylim3d()
    zmin,zmax = ax.get_xlim3d()
    xrange = (xmax - xmin)/2
    yrange = (ymax - ymin)/2
    zrange = (zmax - zmin)/2
    ax.set_xlim3d([origin[0] - xrange, origin[0] + xrange])
    ax.set_ylim3d([origin[1] - yrange, origin[1] + yrange])
    ax.set_zlim3d([origin[2] - zrange, origin[2] + zrange])

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

def plot_pose2_on_axes(axes, pose, axis_length=0.1):
    """
    Plot a 2D pose,  on given axis 'axes' with given 'axis_length'
    is a 2x3 or 3x3 matrix of the form [R | X] 
    where R is 2d rotation and X is translation vector.
    """
    # get rotation and translation (center)
    gRp = pose[:2,:2]  # rotation from pose to global
    origin = pose[:2,-1]

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'g-')

def plot_pose3_on_axes(axes, T, axis_length=0.1, center_plot=False, line_obj_list=None):
    """Plot a 3D pose 4x4 homogenous transform  on given axis 'axes' with given 'axis_length'."""
    return plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length, center_plot, line_obj_list)

def plot_pose3RT_on_axes(axes, gRp, origin, axis_length=0.1, center_plot=False, line_obj_list=None):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    linex = np.append(origin, x_axis, axis=0)
    
    y_axis = origin + gRp[:, 1] * axis_length
    liney = np.append(origin, y_axis, axis=0)

    z_axis = origin + gRp[:, 2] * axis_length
    linez = np.append(origin, z_axis, axis=0)


    if line_obj_list is None:
        xaplt = axes.plot(linex[:, 0], linex[:, 1], linex[:, 2], 'r-')    
        yaplt = axes.plot(liney[:, 0], liney[:, 1], liney[:, 2], 'g-')    
        zaplt = axes.plot(linez[:, 0], linez[:, 1], linez[:, 2], 'b-')
    
        if center_plot:
            center_3d_plot_around_pt(axes,origin[0])
        return [xaplt, yaplt, zaplt]
    
    else:
        line_obj_list[0][0].set_data(linex[:, 0], linex[:, 1])
        line_obj_list[0][0].set_3d_properties(linex[:,2])
        
        line_obj_list[1][0].set_data(liney[:, 0], liney[:, 1])
        line_obj_list[1][0].set_3d_properties(liney[:,2])
        
        line_obj_list[2][0].set_data(linez[:, 0], linez[:, 1])
        line_obj_list[2][0].set_3d_properties(linez[:,2])

        if center_plot:
            center_3d_plot_around_pt(axes,origin[0])
        return line_obj_list

def plot_3d_points(axes, vals, line_obj=None, *args, **kwargs):
    if line_obj is None:
        graph, = axes.plot(vals[:,0], vals[:,1], vals[:,2], *args, **kwargs)
        return graph

    else:
        line_obj.set_data(vals[:,0], vals[:,1])
        line_obj.set_3d_properties(vals[:,2])
        return line_obj

def plot_g2o_SE2(axes, g2o_obj,text=False):
    for key in sorted(g2o_obj.vertices().keys()):
        vert = g2o_obj.vertices()[key]
        print(vert.estimate().to_vector())
        vec = vert.estimate().to_vector()
        R = R2d_from_theta(vec[2])
        t = np.expand_dims(vec[:2],axis=1)
        T = np.vstack((np.hstack((R,t)),np.array([0,0,1])))
        plot_pose2_on_axes(axes,T, axis_length=10.0)
        if text:
            axes.text(t[0,0]+5,t[1,0]+5,str(key))

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


def tiled_features(kp, img_shape, tiley, tilex):
    '''
    Given a set of keypoints, this divides the image into a grid and returns 
    len(kp)/(tilex*tiley) maximum responses within each tell. If that cell doesn't 
    have enough points it will return all of them.
    '''
    feat_per_cell = int(len(kp)/(tilex*tiley))
    HEIGHT, WIDTH = img_shape
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
            inbox_mask = bounding_box(pts, iy, iy+w_height, ix, ix+w_height)
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

'''
def knn_match_and_filter(matcher, kp1, kp2, des1, des2,threshold=0.9):
    matches_knn = matcher.knnMatch(des1,des2, k=2)
    matches = []
    kp1_match = []
    kp2_match = []
    
    for i,match in enumerate(matches_knn):
        if len(match)>1:
            if match[0].distance < threshold*match[1].distance:
                matches.append(match[0])
                kp1_match.append(kp1[match[0].queryIdx].pt)
                kp2_match.append(kp2[match[0].trainIdx].pt)
        elif len(match)==1:
            matches.append(match[0])
            kp1_match.append(kp1[match[0].queryIdx].pt)
            kp2_match.append(kp2[match[0].trainIdx].pt)

    return np.ascontiguousarray(kp1_match), np.ascontiguousarray(kp2_match), matches
'''

def knn_match_and_lowe_ratio_filter(matcher, des1, des2,threshold=0.9):
    # First match 2 against 1
    matches_knn = matcher.knnMatch(des2,des1, k=2)
    
    matches = []
    # Run lowes filter and filter with difference higher than threshold this might
    # sill leave multiple matches into 1 (train descriptors)
    # Create mask of size des1 x des2 for permissible matches
    mask = np.zeros((des1.shape[0],des2.shape[0]),dtype='uint8')
    for match in matches_knn:
        if len(match)==1 or (len(match)>1 and match[0].distance < threshold*match[1].distance):
                matches.append(match[0])
                mask[match[0].trainIdx,match[0].queryIdx] = 1
    
    # run matches again using mask but from 1 to 2 which should remove duplicates            
    matches_cross = matcher.match(des1,des2,mask=mask)
    
    return matches_cross
    '''
    matches_knn = matcher.knnMatch(des1,des2, k=2)
    matches = []
    
    for i,match in enumerate(matches_knn):
        if len(match)>1:
            if match[0].distance < threshold*match[1].distance:
                matches.append(match[0])
        elif len(match)==1:
            matches.append(match[0])
            
    return matches
    '''


'''
def track_keypoints(kp1, des1, kp2, des2, matches):
    
    track_keypoints accepts 2 list of keypoints and descriptors and a list of 
    matches and returns, matched lists of keypoints and descriptors. Also returns
    a set of candidate keypoints from 2 which didn't have matches in 1
    
    Parameters
    ----------
    kp1 : list of cv2.KeyPoint objects
        Keypoints from frame 1
    des1 : NxD array
        NxD array of descriptors where N = len(kp1) and D is length of descriptor
    kp2 : list of cv2.KeyPoint objects
        Keypoints from frame 2
    des2 : NxD array
        NxD array of descriptors where N = len(kp2) and D is length of descriptor        
    matches : list of cv2.DMatch objects
        List of matches between 1 and 2. (single matches only not KNN)
    
    kp1_matched = []    
    kp2_matched = []
    des1_matched = np.zeros((len(matches),des1.shape[1]),des1.dtype)
    des2_matched = np.zeros((len(matches),des2.shape[1]),des2.dtype)
    matched_kp2_indx = []

    # Go through matches and create subsets of kp1 and kp2 which matched
    for i,m in enumerate(matches):
        kp1_matched += [kp1[m.queryIdx]]
        kp2_matched += [kp2[m.trainIdx]]
        des1_matched[i,:] = des1[m.queryIdx,:]
        des2_matched[i,:] = des2[m.trainIdx,:]
        matched_kp2_indx += [m.trainIdx]

    kp2_cand=[]
    cand_indx = []
    for i,k in enumerate(kp2):
        if i not in matched_kp2_indx:
            kp2_cand += [k]
            cand_indx += [i]
    des2_cand = des2[cand_indx,:]
    
    return kp1_matched, des1_matched, kp2_matched, des2_matched, kp2_cand, des2_cand
'''
def track_kp_array(kp1_pts, des1, kp2_pts, des2, matches, lm1=None):
    '''
    track_keypoints accepts 2 arrays of keypoint coordinates and descriptors and a list of 
    matches and returns, matched arrays of keypoints and descriptors. Also returns
    a set of candidate keypoints coords from 2 which didn't have matches in 1
    
    Parameters
    ----------
    kp1 : N1x2 array 
        KeyPoint coordinates from frame 1
    des1 : N1xD array
        N1xD array of descriptors where N1 number of keypoints and D is length of descriptor
    kp2 : N2x2 array
        Keypoint coordinates from frame 2
    des2 : N2xD array
        N2xD array of descriptors where N2 number of keypoints and D is length of descriptor        
    matches : list of cv2.DMatch objects
        List of matches between 1 and 2. (single matches only not KNN)
    '''
    kp1_matched_inds = []    
    kp2_matched_inds = []
    #matched_kp2_indx = []

    # Go through matches and create list of indices of kp1 and kp2 which matched
    for i,m in enumerate(matches):
        kp1_matched_inds += [m.queryIdx]
        kp2_matched_inds += [m.trainIdx]
    
    kp1_match_pts = kp1_pts[kp1_matched_inds,:]
    des1_matched  = des1[kp1_matched_inds,:]
    if lm1 is not None:
        lm1_matched   = lm1[kp1_matched_inds,:]
    kp2_match_pts = kp2_pts[kp2_matched_inds,:]
    des2_matched  = des2[kp2_matched_inds,:]
    
    kp2_cand=[]
    cand_indx = []
    for i in range(kp2_match_pts.shape[0]):
        if i not in kp2_matched_inds:
            #kp2_cand += [k]
            cand_indx += [i]
    kp2_cand_pts = kp2_pts[cand_indx,:]
    des2_cand = des2[cand_indx,:]
    
    return kp1_match_pts, des1_matched, kp2_match_pts, des2_matched, kp2_cand_pts, des2_cand

def propogate_kp(matcher, kp1l, des1l, kp1c, des1c, lm1, kp2, des2, threshold=0.9):
    '''
    track_keypoints accepts 2 arrays of keypoint coordinates and descriptors and a list of 
    matches and returns, matched arrays of keypoints and descriptors. Also returns
    a set of candidate keypoints coords from 2 which didn't have matches in 1
    
    Parameters
    ----------
    matcher : cv2.BFMatcher object
        Matcher to be used for matching keypoint descriptors
    kp1l : Nx2 array 
        KeyPoint coordinates from frame 1 which has associated landmarks in lm1
    des1l : NxD array
        NxD array of descriptors of length D associated with keypoints in kp1l
    kp1c : Nx2 array 
        Candidate KeyPoint coordinates from frame 1 which are not associated with landmarks
    des1c : NxD array
        NxD array of descriptors of length D associated with keypoints in kp1c        
    kp2 : N2x2 array
        Keypoint coordinates from frame 2
    des2 : N2xD array
        N2xD array of descriptors where N2 number of keypoints and D is length of descriptor        
    '''
    vslog = logging.getLogger(__name__)
    if len(kp1l) != len(des1l):
        raise ValueError('Length of kp1l:{} doesn''t match length of des1l:'.format(len(kp1l),len(des1l)))

    if len(kp1l) != len(lm1):
        raise ValueError('Length of kp1l:{} doesn''t match length of lm1:'.format(len(kp1l),len(lm1)))

    if len(kp1c) != len(des1c):
        raise ValueError('Length of kp1c:{} doesn''t match length of des1c:'.format(len(kp1c),len(des1c)))

    if len(kp2) != len(des2):
        raise ValueError('Length of kp2:{} doesn''t match length of des2:'.format(len(kp2),len(des2)))

    #matches for features which were previously tracked and had landmarks
    matches_l = knn_match_and_lowe_ratio_filter(matcher, des1l, des2, threshold) 
    vslog.debug("Found {} matches for existing landmarks out of {}".format(len(matches_l),len(lm1)))
        
    kp1l_matched_inds = []    
    kp2l_matched_inds = []

    # Go through matches and create list of indices of kp1 and kp2 which matched
    for i,m in enumerate(matches_l):
        kp1l_matched_inds += [m.queryIdx]
        kp2l_matched_inds += [m.trainIdx]
    
    kp1l_match = kp1l[kp1l_matched_inds,:]
    des1l_match  = des1l[kp1l_matched_inds,:]
    if lm1 is not None:
        lm1_match   = lm1[kp1l_matched_inds,:]
    kp2l_match = kp2[kp2l_matched_inds,:]
    des2l_match  = des2[kp2l_matched_inds,:]
    
    #matches for features which are candidates from previous image and 
    #not associated with landmarks
    matches_c = knn_match_and_lowe_ratio_filter(matcher, des1c, des2)
    vslog.debug("Found {} matches out of {} prev candidates".format(len(matches_c),len(des1c)))
        
    kp1c_matched_inds = []    
    kp2n_matched_inds = [] # n for new features

    # Go through matches and create list of indices of kp1 and kp2 which matched
    for i,m in enumerate(matches_c):
        kp1c_matched_inds += [m.queryIdx]
        kp2n_matched_inds += [m.trainIdx]
    
    kp1c_match = kp1c[kp1c_matched_inds,:]
    des1c_match  = des1c[kp1c_matched_inds,:]
    kp2n_match = kp2[kp2n_matched_inds,:]
    des2n_match  = des2[kp2n_matched_inds,:]
    
    # Separate out kp2 points which are not matched with either kp1l or kp1c
    kp2c_indx = []
    for i in range(kp2.shape[0]):
        if not (i in kp2l_matched_inds or i in kp2n_matched_inds):
            kp2c_indx += [i]
    kp2c = kp2[kp2c_indx,:]
    des2c = des2[kp2c_indx,:]
    vslog.debug("Number of unmatched candidates: {}".format(len(kp2c)))
    
    vslog.debug("No of kp1l_match: {}, kp1c_match: {}, kp2l_match: {}, kp2n_match: {}, kp2c: {}, des2c: {}".format(
                len(kp1l_match),len(kp1c_match),len(kp2l_match),len(kp2n_match),len(kp2c),len(des2c)))


    return kp1l_match, kp1c_match, kp2l_match, des2l_match, kp2n_match, des2n_match, kp2c, des2c, lm1_match

def combine_and_filter(kp_i_lm, kp_i_cand, kp_j_lm, kp_j_new, K, D, findEssential_set,FILTER_RP=False):
    '''
    Take 2 sets of potentially matching keypoints for landmarks (which have a 3d landmark)
    and candidates (which don't have an associated landmark), stack them into one set
    and run them through a "Esstentially matrix" filter and optionlly "Recover Pose" filter.
    
    Parameters
    ----------
    kp_i_lm : Nx2 array 
        KeyPoint coordinates from frame i which has associated landmarks in lmi
    kp_i_cand : Nx2 array
        Candidate KeyPoint coordinates from frame i
    kp_j_lm : Nx2 array 
        KeyPoint coordinates from frame j which has associated landmarks in lmj
    kp_j_new : Nx2 array
        Candidate KeyPoint coordinates from frame j
    K: 3,3 array
        Camera matrix 3x3
    D: Nx1 array
        Camera distortion coefficients in Opencv format
    findEssential_set: Dict    
        Dictionary of settings for findEssential
    
    Returns
    ----------
    mask_RP_lm : (N,) 1-d numpy boolean array
        Boolean array showing inliners in the landmark points
    mask_RP_cand : (N,) 1-d numpy boolean array
        Boolean array showing inlines in the candidate points
    kp_i_cand_ud: Nx2 array 
        Array of undistorted coordinates of filtered candidate pts in i
    kp_j_new_ud: Nx2 array
        Array of undistorted coordinates of filtered candidate pts in j

    '''    
    vslog = logging.getLogger('vslam')
    if len(kp_i_lm) != len(kp_j_lm):
        raise ValueError('Length of kp_i_lm:{} doesn''t match length of kp_j_lm:'.format(len(kp_i_lm),len(kp_j_lm)))
    else:
        no_landmarks = len(kp_i_lm)
        
    if len(kp_i_cand) != len(kp_j_new):
        raise ValueError('Length of kp_i_cand:{} doesn''t match length of kp_j_new:'.format(len(kp_i_cand),len(kp_j_new)))
    else:
        no_cand = len(kp_i_cand)
    
    kp_i_all = np.vstack((kp_i_lm, kp_i_cand))
    kp_j_all = np.vstack((kp_j_lm, kp_j_new))
    
    kp_i_ud = cv2.undistortPoints(np.expand_dims(kp_i_all,1),K,D)[:,0,:]
    kp_j_ud = cv2.undistortPoints(np.expand_dims(kp_j_all,1),K,D)[:,0,:]

    E, mask_e_all = cv2.findEssentialMat(kp_i_ud, kp_j_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, **findEssential_set)
    essen_mat_pts = np.sum(mask_e_all)  
    
    vslog.debug("Essential matrix: used {} of total {} matches".format(essen_mat_pts,len(kp_j_all)))
    
    if FILTER_RP:
        # Recover Pose filtering is breaking under certain conditions. Leave out for now.
        _, _, _, mask_RP_all = cv2.recoverPose(E, kp_i_ud, kp_j_ud, mask=mask_e_all)
        vslog.info("Recover pose: used {} of total {} matches".format(np.sum(mask_RP_all),essen_mat_pts))
    else: 
        mask_RP_all = mask_e_all
    
    # Split the combined mask to lm features and candidates
    mask_RP_lm = mask_RP_all[:no_landmarks,0].astype(bool)
    mask_RP_cand = mask_RP_all[-no_cand:,0].astype(bool)
    
    kp_i_cand_ud = kp_i_ud[-no_cand:]
    kp_j_new_ud  = kp_j_ud[-no_cand:]
    
    kp_i_cand_ud = kp_i_cand_ud[mask_RP_cand]
    kp_j_new_ud  = kp_j_new_ud[mask_RP_cand]
    
    return mask_RP_lm, mask_RP_cand, kp_i_cand_ud, kp_j_new_ud

def trim_using_mask(mask, *argv):
    trimmed_arr = []
    for arr in argv: 
        trimmed_arr.append(arr[mask[:,0].astype(bool)])
    return trimmed_arr

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
    ''' 
    Given a set of Keypoints this finds the maximum response within radial 
    distance from each other 
    '''
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

def remove_redundant_newkps(kp_new, kp_old, dist):
    old_feat_pts = kp_old[:,0,:]
    tree = spatial.cKDTree(old_feat_pts)
    new_feat_pts = kp_new[:,0,:]
    idx = tree.query_ball_point(new_feat_pts, dist, p=2., eps=0, n_jobs=1)
    newpt_mask = idx.astype(bool)
    return kp_new[~newpt_mask]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))