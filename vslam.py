#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from vslam_helper import *
from ssc import *
np.set_printoptions(precision=3,suppress=True)
 
print (sys.platform)

# Inputs, images and camera info

CHESSBOARD = True
if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
else:
    path = '/home/vik748/'
img1 = cv2.imread(path+'data/chess_board2/GOPR1496.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board2/GOPR1497.JPG',1)  
img3 = cv2.imread(path+'data/chess_board2/GOPR1498.JPG',1)  

mask_pts_1488 = np.array([[1180, 960], [2740, 1000], [2700, 2040], [1180, 1980]])
mask_pts_1489 = np.array([[1180, 1225], [2550, 1475], [2400, 2340], [1100, 2100]])
mask_pts_1490 = np.array([[1550, 1250], [2870, 1030], [2900, 1890], [1680, 2200]])
mask_pts_1491 = np.array([[760, 880], [2100, 1090], [2180, 1980], [780, 2100]])
mask_pts_1496 = np.array([[400, 650], [1300, 680], [1280, 1300], [390, 1260]])
mask_pts_1497 = np.array([[760, 1055], [1600, 1080], [1560, 1670], [755, 1630]])
mask_pts_1498 = np.array([[1030, 740], [1820, 840], [1785, 1085], [1675, 1400], [975, 1285]])

mask = np.zeros(img1.shape[:2], dtype=np.uint8)

mask1 = 255 - cv2.fillConvexPoly(mask, mask_pts_1496, color=[255, 255, 255])
mask2 = 255 - cv2.fillConvexPoly(mask, mask_pts_1497, color=[255, 255, 255])
mask3 = 255 - cv2.fillConvexPoly(mask, mask_pts_1498, color=[255, 255, 255])
'''
img1 = cv2.imread(path+'data/chess_board2/GOPR1496.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board2/GOPR1497.JPG',1)  
img3 = cv2.imread(path+'data/chess_board2/GOPR1498.JPG',1)
'''

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gr3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

# create a mask image filled with zeros, the size of original image

'''
fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])
'''    
K = np.array([[3.50275628e+03, 0.00000000e+00, 2.01997668e+03],
              [0.00000000e+00, 3.47709480e+03, 1.44976175e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
D = np.array([[-2.85711277e-01,  1.61304120e-01,  5.36070359e-05, -1.48554708e-04,
               -7.71783829e-02]])

print(K,D)

#Initiate ORB detector
detector = cv2.ORB_create(nfeatures=25000, edgeThreshold=65, patchSize=65, nlevels=4, 
                     fastThreshold=10, scaleFactor=10.0, WTA_K=4,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)
                     
#detector = cv2.AKAZE_create(threshold=.0005)
#detector = cv2.BRISK_create(thresh = 15, octaves = 10, patternScale = 1.0 )
# find the keypoints and descriptors with ORB
kp1 = detector.detect(gr1,mask1)
kp2 = detector.detect(gr2,mask2)
kp3 = detector.detect(gr3,mask3)
print ("Points detected: ",len(kp1))
'''
kp1 = sorted(kp1, key = lambda x:x.response, reverse = True)
kp2 = sorted(kp2, key = lambda x:x.response, reverse = True)
kp3 = sorted(kp3, key = lambda x:x.response, reverse = True)
print ("Points sorted: ")

kp1 = SSC(kp1, 10000, 0.1, gr1.shape[1], gr1.shape[0])
kp2 = SSC(kp2, 10000, 0.1, gr1.shape[1], gr1.shape[0])
kp3 = SSC(kp3, 10000, 0.1, gr1.shape[1], gr1.shape[0])
print ("Points nonmax supression: ")
'''
kp1 = radial_non_max(kp1,25)
kp2 = radial_non_max(kp2,25)
kp3 = radial_non_max(kp3,25)

kp1, des1 = detector.compute(gr1,kp1)
kp2, des2 = detector.compute(gr2,kp2)
kp3, des3 = detector.compute(gr3,kp3)
print ("Descriptors computed: ")

# create BFMatcher object - Brute Force
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#FLANN_INDEX_LSH = 6
#FLANN_INDEX_KDTREE = 1
#matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 20,
#                                   multi_probe_level = 2),
#                              dict(checks=100))

# Match descriptors.
'''
kp1_match_12, kp2_match_12, matches12 = knn_match_and_filter(matcher, kp1, kp2, des1, des2)
kp2_match_23, kp3_match_23, matches23 = knn_match_and_filter(matcher, kp2, kp3, des2, des3)

'''
matches12 = matcher.match(des1,des2)
matches23 = matcher.match(des2,des3)
kp1_match_12 = np.array([kp1[mat.queryIdx].pt for mat in matches12])
kp2_match_12 = np.array([kp2[mat.trainIdx].pt for mat in matches12])

kp2_match_23 = np.array([kp2[mat.queryIdx].pt for mat in matches23])
kp3_match_23 = np.array([kp3[mat.trainIdx].pt for mat in matches23])

#matches12 = sorted(matches12, key = lambda x:x.distance)
#matches12 = matches12[:(int)(len(matches12)*.75)]

kp1_match_12_ud = cv2.undistortPoints(np.expand_dims(kp1_match_12,axis=1),K,D)
kp2_match_12_ud = cv2.undistortPoints(np.expand_dims(kp2_match_12,axis=1),K,D)

E_12, mask_e_12 = cv2.findEssentialMat(kp1_match_12_ud, kp2_match_12_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e_12) ," of total ",len(matches12),"matches")

points, R_21, t_21, mask_RP_12 = cv2.recoverPose(E_12, kp1_match_12_ud, kp2_match_12_ud,mask=mask_e_12)
T_2_1 = compose_T(R_21,t_21)
T_1_2 = T_inv(T_2_1)
print("points:",points,"\trecover pose mask:",np.sum(mask_RP_12!=0))
print("R:",R_21)
print("t:",t_21.T)

img12 = displayMatches(gr1,kp1,gr2,kp2,matches12,mask_RP_12, False)
fig1 = plt.figure(1)
#plt.get_current_fig_manager().window.setGeometry(0, 0, 928, 1028)
move_figure(position="left")
plt.imshow(img12),
plt.ion()
#plt.show()
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

landmarks_12 = triangulate(np.eye(4), T_1_2, kp1_match_12_ud[mask_RP_12[:,0]==1], 
                                             kp2_match_12_ud[mask_RP_12[:,0]==1])

fig = plt.figure(2)
ax2 = fig.add_subplot(111, projection='3d')
#plt.get_current_fig_manager().window.setGeometry(992, 430, 928, 1028)
move_figure(position="right")
ax2.set_aspect('equal')         # important!
title = ax2.set_title('After triangulation with 1 and 2')
graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker="o")

if CHESSBOARD:
    ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
    ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)
    ret3, corners3 = cv2.findChessboardCorners(gr3, (16,9),None)
    
    corners1_ud = cv2.undistortPoints(corners1,K,D)
    corners2_ud = cv2.undistortPoints(corners2,K,D)
    corners3_ud = cv2.undistortPoints(corners3,K,D)
   
    corners_12 = triangulate(np.eye(4), T_1_2, corners1_ud, corners2_ud)
    graph = plot_3d_points(ax2, corners_12, linestyle="", marker=".",color='g')

plot_pose3_on_axes(ax2, T_1_2, axis_length=1.0)
plot_pose3_on_axes(ax2,np.eye(4), axis_length=0.5)

set_axes_equal(ax2)
ax2.view_init(-70, -90)

plt.draw()
plt.pause(.001)
input("Press [enter] to continue.")
'''
process frame
where the state of a frame at time t , , contains the following data:
S: A set of 2D keypoints {p } (each one of them is associated to a 3D landmark
k k=1..K
The associated set of 3D landmarks {X } .i
'''

lm = -np.ones(mask_RP_12.shape[0],dtype=int)
lm[mask_RP_12.ravel()==1]=np.arange(np.sum(mask_RP_12))

# Create a dictionary {KP2 index of match : landmark number}
frame2_to_lm = {mat.trainIdx:lm_id for lm_id,mat in zip(lm, matches12)
                if lm_id!=-1 }

kp2_match_23_ud = cv2.undistortPoints(np.expand_dims(kp2_match_23,axis=1),K,D)
kp3_match_23_ud = cv2.undistortPoints(np.expand_dims(kp3_match_23,axis=1),K,D)

E_23, mask_e_23 = cv2.findEssentialMat(kp2_match_23_ud, kp3_match_23_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e_23) ," of total ",len(matches23),"matches")
#mask_RP_23 = mask_e_23
points, R_32, t_32, mask_RP_23 = cv2.recoverPose(E_23, kp2_match_23_ud, kp3_match_23_ud,mask=mask_e_23)
img23 = displayMatches(gr2,kp2,gr3,kp3,matches23,mask_RP_23, False)

fig1 = plt.figure(1)
plt.imshow(img23)
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

matches23_filt = [matches23[i] for i in range(len(matches23)) if mask_RP_23[i]==1]

frame3_to_frame2 = {mat.trainIdx:mat.queryIdx for mat in matches23_filt}

frame3_to_lm = {id:frame2_to_lm.get(frame3_to_frame2[id]) 
                for id in frame3_to_frame2.keys() 
                if frame2_to_lm.get(frame3_to_frame2[id]) is not None}

print("Frame3_to_lm: ",len(frame3_to_lm))

landmarks_23 = np.array([landmarks_12[frame3_to_lm[k]] for k in 
                        frame3_to_lm.keys()])

lm_kps_3 = np.array([kp3[k].pt for k in frame3_to_lm.keys()])
success, T_2_3, inliers = T_from_PNP(landmarks_23, lm_kps_3, K, D)

landmarks_23_new = triangulate(T_1_2, T_2_3, kp2_match_23_ud[mask_RP_23[:,0]==1], 
                                             kp3_match_23_ud[mask_RP_23[:,0]==1])

plt.figure(2)
graph = plot_3d_points(ax2, landmarks_23, linestyle="", marker="o", color='r')
plot_pose3_on_axes(ax2, T_2_3, axis_length=2.0)


if CHESSBOARD:
    corners_23 = triangulate(T_1_2, T_2_3, corners2_ud, corners3_ud)
    graph = plot_3d_points(ax2, corners_23, linestyle="", marker=".",color='tab:orange')

set_axes_equal(ax2)             # important!
plt.draw()
plt.pause(0.01)
input("Press [enter] to continue.")
landmarks_23_new = triangulate(T_1_2, T_2_3, kp2_match_23_ud[mask_RP_23[:,0]==1], 
                                             kp3_match_23_ud[mask_RP_23[:,0]==1])
graph = plot_3d_points(ax2, landmarks_23_new, linestyle="", marker="o", color='g')
set_axes_equal(ax2)             # important!
plt.draw()
plt.pause(0.01)
input("Press [enter] to continue.")
plt.close(fig='all')