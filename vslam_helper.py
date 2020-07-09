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
import itertools

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
    return cv2.undistortPoints(np.expand_dims(kps, axis=1),
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


def draw_matches_vertical(img_top, kp1,img_bottom,kp2, matches, mask, display_invalid=False, color=(0, 255, 0)):
    '''
    This function takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    assert img_top.shape == img_bottom.shape
    out_img = np.vstack((img_top, img_bottom))
    bool_mask = mask.astype(bool)
    valid_bottom_matches = np.array([kp2[mat.trainIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    valid_top_matches = np.array([kp1[mat.queryIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    img_height = img_top.shape[0]

    if len(out_img.shape) == 2: out_img = cv2.cvtColor(out_img,cv2.COLOR_GRAY2RGB)

    for p1,p2 in zip(valid_top_matches, valid_bottom_matches):
        cv2.line(out_img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1]+img_height)), color=color, thickness=1)
    return out_img


def draw_points(vis_orig, points, color = (0, 255, 0), thick = 3):
    if vis_orig.shape[2] == 3: vis = vis_orig
    else: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    rad = int(vis.shape[1]/200)
    thick = round(vis.shape[1]/1000)

    for pt in points:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), rad , color, thickness=thick)
    return vis

def draw_markers(vis_orig, keypoints, color = (0, 0, 255)):
    if vis_orig.shape[2] == 3: vis = vis_orig
    else: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)

    for kp in keypoints:
        x, y = kp.pt
        cv2.drawMarker(vis, (int(x), int(y)), color,  markerSize=30, markerType = cv2.MARKER_CROSS, thickness=2)
    return vis


def draw_arrows(vis_orig, points1, points2, color = (0, 255, 0), thick = 2, tip_length = 0.25):
    if len(vis_orig.shape) == 2: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    else: vis = vis_orig
    for p1,p2 in zip(points1,points2):
        cv2.arrowedLine(vis, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),
                        color=color, thickness=thick, tipLength = tip_length)
    return vis

def draw_feature_tracks(img_left,kp1,img_right,kp2, matches, mask, display_invalid=False, color=(0, 255, 0), thick = 2):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    bool_mask = mask.astype(bool)
    valid_right_matches = np.array([kp2[mat.trainIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    valid_left_matches = np.array([kp1[mat.queryIdx].pt for is_match, mat in zip(bool_mask, matches) if is_match])
    #img_right_out = draw_points(img_right, valid_right_matches)
    img_right_out = draw_arrows(img_right, valid_left_matches, valid_right_matches, thick = thick)


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
    img_right_out = draw_arrows(img_right, valid_left_matches, valid_right_matches,
                                color=color, thick = round(img_right.shape[1]/1000))

    return img_right_out

def center_3d_plot_around_pt(ax, origin, zoom_to_fit = False):
    xmin,xmax = ax.get_xlim3d()
    ymin,ymax = ax.get_ylim3d()
    zmin,zmax = ax.get_xlim3d()
    if zoom_to_fit:
        xrange = (xmax - xmin)
        yrange = (ymax - ymin)
        zrange = (zmax - zmin)
    else:
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

def set_axes_equal(ax, limits=None):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    if limits is None:
        limits = np.array([ ax.get_xlim3d(),
                            ax.get_ylim3d(),
                            ax.get_zlim3d()  ])

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

def plot_pose3_on_axes(axes, T, axis_length=0.1, center_plot=False, line_obj_list=None, zoom_to_fit=False):
    """Plot a 3D pose 4x4 homogenous transform  on given axis 'axes' with given 'axis_length'."""
    return plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length, center_plot, line_obj_list, zoom_to_fit=zoom_to_fit)

def plot_pose3RT_on_axes(axes, gRp, origin, axis_length=0.1, center_plot=False, line_obj_list=None, zoom_to_fit=False):
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
            center_3d_plot_around_pt(axes,origin[0],zoom_to_fit=zoom_to_fit)
        return [xaplt, yaplt, zaplt]

    else:
        line_obj_list[0][0].set_data(linex[:, 0], linex[:, 1])
        line_obj_list[0][0].set_3d_properties(linex[:,2])

        line_obj_list[1][0].set_data(liney[:, 0], liney[:, 1])
        line_obj_list[1][0].set_3d_properties(liney[:,2])

        line_obj_list[2][0].set_data(linez[:, 0], linez[:, 1])
        line_obj_list[2][0].set_3d_properties(linez[:,2])

        if center_plot:
            center_3d_plot_around_pt(axes,origin[0],zoom_to_fit=zoom_to_fit)
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


def tiled_features(kp, img_shape, tiles_hor, tiles_ver, no_features = None):
    '''
    Given a set of keypoints, this divides the image into a grid and returns
    len(kp)/(tiles_ver*tiles_hor) maximum responses within each tell. If that cell doesn't
    have enough points it will return all of them.
    '''
    if no_features:
        feat_per_cell = np.ceil(no_features/(tiles_ver*tiles_hor)).astype(int)
    else:
        feat_per_cell = np.ceil(len(kp)/(tiles_ver*tiles_hor)).astype(int)
    HEIGHT, WIDTH = img_shape
    assert WIDTH%tiles_hor == 0, "Width is not a multiple of tiles_ver"
    assert HEIGHT%tiles_ver == 0, "Height is not a multiple of tiles_hor"
    w_width = int(WIDTH/tiles_hor)
    w_height = int(HEIGHT/tiles_ver)

    kps = np.array([])
    #pts = np.array([keypoint.pt for keypoint in kp])
    pts = cv2.KeyPoint_convert(kp)
    kp = np.array(kp)

    #img_keypoints = draw_markers( cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB), kp, color = ( 0, 255, 0 ))


    for ix in range(0,HEIGHT, w_height):
        for iy in range(0,WIDTH, w_width):
            inbox_mask = bounding_box(pts, iy, iy+w_height, ix, ix+w_height)
            inbox = kp[inbox_mask]
            inbox_sorted = sorted(inbox, key = lambda x:x.response, reverse = True)
            inbox_sorted_out = inbox_sorted[:feat_per_cell]
            kps = np.append(kps,inbox_sorted_out)

            #img_keypoints = draw_markers(img_keypoints, kps.tolist(), color = [255, 0, 0] )
            #cv2.imshow("Selected Keypoints", img_keypoints )
            #print("Size of Tiled Keypoints: " ,len(kps))
            #cv2.waitKey();
    return kps.tolist()

'''
def tiled_features2(kp_list, img_shape, tiles_hor, tiles_ver, no_features = None):
    rows = img_shape[0]
    cols = img_shape[1]
    if no_features:
        feat_per_cell = int(no_features/(tiles_hor*tiles_ver))
    else:
        feat_per_cell = int(len(kp_list)/(tiles_hor*tiles_ver))

    assert cols%tiles_hor == 0, "Width is not a multiple of tiles_hor"
    assert rows%tiles_ver == 0, "Height is not a multiple of tiles_ver"

    tile_width = int(cols/tiles_hor)
    tile_height = int(rows/tiles_ver)

    kp_xy_arr = cv2.KeyPoint_convert(kp_list)
    tiled_keypoints = []
    unsel_keypoints = []
    #img_keypoints = draw_markers( cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB), kp_list, color = ( 0, 255, 0 ))

    for min_row in range(0, rows,tile_height):
        for min_col in range(0, cols, tile_width):
            max_row = min_row + tile_height
            max_col = min_col + tile_width

            sel_keypoints = []

            for i, (kp, kp_xy) in enumerate(zip(kp_list, kp_xy_arr)):
                if (kp_xy[0] > min_col and kp_xy[0] <= max_col and
                    kp_xy[1] > min_row and kp_xy[1] <= max_row ):

                        sel_keypoints.append(kp)
                        #keypoints.erase(next(begin(keypoints), i));

            #print("Keypoints in BB: ", len(sel_keypoints))

            #draw_markers( img_keypoints, sel_keypoints, img_keypoints, ( 0, 255, 0 ) );
            #img_keypoints = draw_markers(img_keypoints, cv2.COLOR_GRAY2RGB), sel_keypoints, color=[255,255,0])
            #cv2.imshow("Selected Keypoints", img_keypoints );
            #cv2.waitKey();

            if len(sel_keypoints) > feat_per_cell:
                sel_keypoints = sorted(sel_keypoints, key = lambda x:x.response, reverse = True)
                tiled_keypoints.extend(sel_keypoints[:feat_per_cell])
                unsel_keypoints.extend(sel_keypoints[feat_per_cell:])
            else:
                tiled_keypoints.extend(sel_keypoints)

            #img_keypoints = draw_markers(img_keypoints, tiled_keypoints, color = [255, 0, 0] )
            #cv2.imshow("Selected Keypoints", img_keypoints )
            #print("Size of Tiled Keypoints: " ,len(tiled_keypoints), "Size of unsel keypoints: ", len(unsel_keypoints))
            #cv2.waitKey();


    if len(tiled_keypoints) < no_features:
        unsel_keypoints = sorted(unsel_keypoints, key = lambda x:x.response, reverse = True)
        tiled_keypoints.extend(unsel_keypoints[:(no_features - len(tiled_keypoints))])

    #img_keypoints = draw_markers(img_keypoints, tiled_keypoints, color = [255, 0, 0] )
    #cv2.imshow("Selected Keypoints", img_keypoints )
    #print("Size of Tiled Keypoints: ", len(tiled_keypoints), " Size of unsel keypoints: ", len(unsel_keypoints))
    #cv2.waitKey()
'''

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 15, color)
    return vis

def knn_match_and_lowe_ratio_filter(matcher, des1, des2,threshold=0.9, dist_mask_12=None, draw_plot_dist=False):
    # First match 2 against 1
    if dist_mask_12 is None:
        dist_mask_21 = None
    else:
        dist_mask_21 = dist_mask_12.T
    matches_knn = matcher.knnMatch(des2,des1, k=2, mask = dist_mask_21 )
    all_ds = [m[0].distance for m in matches_knn if len(m) >0]

    #print("Len of knn matches", len(matches_knn))

    matches = []
    # Run lowes filter and filter with difference higher than threshold this might
    # still leave multiple matches into 1 (train descriptors)
    # Create mask of size des1 x des2 for permissible matches
    mask = np.zeros((des1.shape[0],des2.shape[0]),dtype='uint8')
    for match in matches_knn:
        if len(match)==1 or (len(match)>1 and match[0].distance < threshold*match[1].distance):
           # if match[0].distance < 75:
                matches.append(match[0])
                mask[match[0].trainIdx,match[0].queryIdx] = 1

    #matches = [m for m in matches if m.distance<5 ]

    if draw_plot_dist:
        fig, axes = plt.subplots(1, 1, num=3)
        filt_ds = [m.distance for m in matches]
        axes.plot(sorted(all_ds),'.',label = 'All Matches')
        axes.plot(sorted(filt_ds),'.',label = 'Filtered Matches')
        axes.set_xlabel('Number')
        axes.set_ylabel('Distance')
        axes.legend()
        plt.pause(.1)

    # run matches again using mask but from 1 to 2 which should remove duplicates
    # This is basically same as running cross match after lowe ratio test
    matches_cross = matcher.match(des1,des2,mask=mask)
    #print("Len of cross matches", len(matches_cross))
    return matches_cross

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

def radial_non_max_kd(kp_list, dist):
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



def radial_non_max(kp_list, image_size, kernel_size = (3,3)):
    '''
    Given a set of Keypoints this finds the maximum response within radial
    distance from each other
    '''
    resp = np.zeros(image_size)
    kp_map = np.zeros(image_size,dtype=int)
    non_max_kernel = np.ones(kernel_size, np.uint8)
    kp_xy = np.rint(cv2.KeyPoint_convert(kp_list)).astype(int)

    #print ("len of kp1:",len(kp))
    for i, (k, kp_loc) in enumerate(zip(kp_list,kp_xy)):
        #kp_loc = np.rint(k.pt).astype(int)
        kp_map[kp_loc[1], kp_loc[0]] = i
        resp[kp_loc[1], kp_loc[0]] = k.response

    non_max_locs = cv2.dilate(resp, non_max_kernel, iterations=1) > resp
    kp_ind = kp_map[np.logical_and(kp_map,~non_max_locs)]

    return [kp_list[i] for i in kp_ind]

def keypoint_distance_search_mask_opencv(kp1_pts, kp2_pts, dist):
     '''
     Same as keypoint_distance_search_mask but using OpenCV's matcher
     Given a 2 sets of keypoints create a 2d mask where mask[i,j] is True if i-th
     element in kp1 is within dist from j-th element in kp2
     '''

     bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
     matches2d = bf_matcher.radiusMatch(kp1_pts,kp2_pts, maxDistance = dist)
     mask = np.zeros((len(kp1_pts),len(kp2_pts)),dtype='uint8')
     for i, matches in enumerate(matches2d):
         idx = [m.trainIdx for m in matches]
         mask[i][idx] = 1
     return mask


def keypoint_distance_search_mask(kp1_pts, kp2_pts, dist, kp1_tree=None, kp2_tree=None):
    '''
    Given a 2 sets of keypoints create a 2d mask where mask[i,j] is True if i-th
    element in kp1 is within dist from j-th element in kp2
    '''
    mask = np.zeros((len(kp1_pts),len(kp2_pts)),dtype='uint8')

    if kp2_tree is None:
        kp2_tree = spatial.cKDTree(kp2_pts)
    '''
    for i,k1 in enumerate(kp1_pts):
        mask[i,kp2_tree.query_ball_point(k1, dist, p=2., eps=0, n_jobs=1)] = True
    '''
    if kp1_tree is None:
        kp1_tree = spatial.cKDTree(kp1_pts)
    inliers_2dlist = kp1_tree.query_ball_tree(kp2_tree, dist, p=2)
    '''
    for i, r in enumerate(inliers_2dlist):
        mask[i][r] = True

    The following is a little optimized version of above
    '''
    sz = np.fromiter(map(len,inliers_2dlist),int,len(kp1_pts))
    mask[np.arange(len(kp1_pts)).repeat(sz),np.fromiter(itertools.chain.from_iterable(inliers_2dlist),int,sz.sum())] = 1

    return mask, kp1_tree, kp2_tree

    '''
    Super optimized: https://stackoverflow.com/questions/56660387/fastest-way-to-convert-a-list-of-indices-to-2d-numpy-array-of-ones
        %load_ext cython


        %%cython

        cimport cython
        cimport numpy as cnp
        import numpy as np

        @cython.boundscheck(False)  # remove this if you cannot guarantee that nrow/ncol are correct
        @cython.wraparound(False)
        cpdef cnp.int_t[:, :] mseifert(list a, int nrow, int ncol):
            cdef cnp.int_t[:, :] out = np.zeros([nrow, ncol], dtype=int)
            cdef list subl
            cdef int row_idx
            cdef int col_idx
            for row_idx, subl in enumerate(a):
                for col_idx in subl:
                    out[row_idx, col_idx] = 1
            return out
    '''

def match_image_pairs(detector, descriptor, imgs, K=np.eye(3), D=np.zeros(4),
                      tiling=None, pixel_matching_dist = None, lowe_threshold=0.9,
                      feat_img_axes = None, matching_img_axes = None, match_dist_plot_axes = None ):
    '''
    Given a pair of images, a feature detector and a descriptor, display the
    matches after filtering with Essential matrix filter
    '''

    if detector == descriptor:
        kp_0, des_0 = detector.detectAndCompute(imgs[0], mask=None)
        kp_1, des_1 = detector.detectAndCompute(imgs[1], mask=None)

    else:
        kp_0 = detector.detect(imgs[0], mask=None)
        kp_1 = detector.detect(imgs[1], mask=None)

        print ("Points before tiling supression: {} and {} ".format(len(kp_0),len(kp_1)))

        if not (tiling is None) :
            kp_0 = tiled_features(kp_0, imgs[0].shape, tiling['x'], tiling['y'], no_features = tiling['no_features'])
            kp_1 = tiled_features(kp_1, imgs[1].shape, tiling['x'], tiling['y'], no_features = tiling['no_features'])
            print ("Points after tiling supression: {} and {} ".format(len(kp_0),len(kp_1)))

        kp_0, des_0 = descriptor.compute(imgs[0], kp_0)
        kp_1, des_1 = descriptor.compute(imgs[1], kp_1)

    kp_0_sort = sorted(kp_0, key = lambda x: x.response, reverse=True)
    kp_1_sort = sorted(kp_1, key = lambda x: x.response, reverse=True)

    kp_img_0 = draw_markers (cv2.cvtColor(imgs[0], cv2.COLOR_GRAY2RGB), kp_0 ,color=[255,255,0])
    kp_img_1 = draw_markers (cv2.cvtColor(imgs[1], cv2.COLOR_GRAY2RGB), kp_1 ,color=[255,255,0])


    if not (feat_img_axes is None):
        feat_img_axes.axis("off");
        feat_img_axes.set_title("Det: {}".format(detector.getDefaultName()))
        feat_img_axes.imshow(kp_img_0)

    #     Match and find inliers
    if descriptor.getDefaultName() == 'Feature2D.ORB' and descriptor.getWTA_K() != 2:
        print ("SETTING HAMMING2")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=False)

    kp_0_pts = cv2.KeyPoint_convert(kp_0)
    kp_1_pts = cv2.KeyPoint_convert(kp_1)

    if pixel_matching_dist is not None:
        dist_mask,_ ,_ = keypoint_distance_search_mask(kp_0_pts, kp_1_pts, pixel_matching_dist)
    else:
        dist_mask = None

    matches_01 = knn_match_and_lowe_ratio_filter(matcher, des_0, des_1, threshold=lowe_threshold,
                                                 dist_mask_12 = dist_mask)

    kp0_match_01 = np.array([kp_0[mat.queryIdx].pt for mat in matches_01])
    kp1_match_01 = np.array([kp_1[mat.trainIdx].pt for mat in matches_01])

    kp0_match_01_ud = cv2.undistortPoints(np.expand_dims(kp0_match_01,axis=1),K,D)
    kp1_match_01_ud = cv2.undistortPoints(np.expand_dims(kp1_match_01,axis=1),K,D)

    E_12, mask_e_12 = cv2.findEssentialMat(kp0_match_01_ud, kp1_match_01_ud, focal=1.0, pp=(0., 0.),
                                           method=cv2.RANSAC, prob=0.9999, threshold=0.0004)

    #plt.figure(3)
    #plt.plot(np.linalg.norm(kp0_match_01[mask_e_12[:,0]==1] - kp1_match_01[mask_e_12[:,0]==1], axis = 1),'.')

    if not (match_dist_plot_axes is None):
        matches_01_dist = np.array([mat.distance for mat in matches_01])
        match_dist_plot_axes.plot(np.sort(matches_01_dist),'.',label = 'Filt: Lowe ratio + cross matches')
        match_dist_plot_axes.plot(np.sort(matches_01_dist[mask_e_12[:,0]==1]),'.',label = 'Essential mat inliers')
        match_dist_plot_axes.set_xlabel('Index number')
        match_dist_plot_axes.set_ylabel('Distance')
        match_dist_plot_axes.legend()

    #essn_ds = [m.distance for m, mv in zip(matches_01, mask_e_12) if mv ==1 ]

    #fig, axes = plt.subplots(1, 1, num=3)
    #plt.legend()
    #plt.pause(.1)


    print("After essential mat ransac: {} / {} ".format(np.sum(mask_e_12), len(matches_01)))

    valid_matches_img = draw_matches_vertical(imgs[0], kp_0, imgs[1], kp_1, matches_01,
                                              mask_e_12, display_invalid=True, color=(0, 255, 0))

    if not (matching_img_axes is None):
        print("matching_img_axes")
        matching_img_axes.axis("off")
        matching_img_axes.set_title("Des: {}\n{:d} matches".format(descriptor.getDefaultName(),np.sum(mask_e_12)))
        matching_img_axes.imshow(valid_matches_img)

    return kp_img_0, valid_matches_img

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
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_distance(Ra, Rb):
    """ Returns the angle between two rotation matrices"""
    R = Ra @ Rb.T
    return np.rad2deg(np.arccos((np.trace(R) - 1) / 2))

def read_metashape_poses(file):
    """ Read a text file where each line has image name followed by the 4x4 pose as
        1 x 16 row and return it as a dictionary """
    pose_dict = {}
    with open(file) as f:
        first_line = f.readline()
        if not first_line.startswith('Image_name,4x4 Tmatrix as 1x16 row'):
            raise ValueError("File doesn't start with 'Image_name,4x4 Tmatrix as 1x16 row' might be wrong format")
        data = f.readlines()
        for i,line in enumerate(data):
            name, T_string = (line.strip().split(',',maxsplit=1))
            T = np.fromstring(T_string,sep=',').reshape((4,4))
            pose_dict[name] = T
    return pose_dict