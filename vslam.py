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
img1 = cv2.imread(path+'data/chess_board/GOPR1488.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board/GOPR1490.JPG',1)  
img3 = cv2.imread(path+'data/chess_board/GOPR1491.JPG',1)  

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gr3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

# create a mask image filled with zeros, the size of original image
mask = np.zeros(img1.shape[:2], dtype=np.uint8)

mask_pts_1488 = np.array([[1180, 960], [2740, 1000], [2700, 2040], [1180, 1980]])
mask_pts_1489 = np.array([[1180, 1225], [2550, 1475], [2400, 2340], [1100, 2100]])
mask_pts_1490 = np.array([[1550, 1250], [2870, 1030], [2900, 1890], [1680, 2200]])
mask_pts_1491 = np.array([[760, 880], [2100, 1090], [2180, 1980], [780, 2100]])

mask1 = 255 - cv2.fillConvexPoly(mask, mask_pts_1488, color=[255, 255, 255])
mask2 = 255 - cv2.fillConvexPoly(mask, mask_pts_1490, color=[255, 255, 255])
mask3 = 255 - cv2.fillConvexPoly(mask, mask_pts_1491, color=[255, 255, 255])

fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])

D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);

print(K,D)


#Initiate ORB detector
detector = cv2.ORB_create(nfeatures=int(25000), edgeThreshold=32, patchSize=32, nlevels=16, 
                     fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=4)
                     
#detector = cv2.AKAZE_create(threshold=.0005)
#detector = cv2.BRISK_create(thresh = 15, octaves = 10, patternScale = 1.0 )

# find the keypoints and descriptors with ORB
kp1 = detector.detect(gr1,mask1)
kp2 = detector.detect(gr2,mask2)
kp3 = detector.detect(gr3,mask3)
'''
print ("Points detected: ",len(kp1))

kp1 = sorted(kp1, key = lambda x:x.response, reverse = True)
kp2 = sorted(kp2, key = lambda x:x.response, reverse = True)
kp3 = sorted(kp3, key = lambda x:x.response, reverse = True)

print ("Points sorted: ")

kp1 = SSC(kp1, 5000, 0.1, gr1.shape[1], gr1.shape[0])
kp2 = SSC(kp2, 5000, 0.1, gr1.shape[1], gr1.shape[0])
kp3 = SSC(kp3, 5000, 0.1, gr1.shape[1], gr1.shape[0])

print ("Points nonmax supression: ")
'''
kp1, des1 = detector.compute(gr1,kp1)
kp2, des2 = detector.compute(gr2,kp2)
kp3, des3 = detector.compute(gr3,kp3)

print ("Descriptors computed: ")

# create BFMatcher object - Brute Force
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#FLANN_INDEX_LSH = 6
#FLANN_INDEX_KDTREE = 1
#matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 20,
#                                   multi_probe_level = 2),
#                              dict(checks=100))

# Match descriptors.
matches12_knn = matcher.knnMatch(des1,des2, k=2)
matches23_knn = matcher.knnMatch(des2,des3, k=2)

#matches12 = sorted(matches12, key = lambda x:x.distance)
#matches12 = matches12[:(int)(len(matches12)*.75)]


#matches = sorted(matches, key = lambda x:x.distance)

#kp1_match_12 = np.array([kp1[mat.queryIdx].pt for mat in matches12])
#kp2_match_12 = np.array([kp2[mat.trainIdx].pt for mat in matches12])

matches12 = []
kp1_match_12 = []
kp2_match_12 = []

for i,match in enumerate(matches12_knn):
    if len(match)>1:
        if match[0].distance < 0.95*match[1].distance:
            matches12.append(match[0])
            kp1_match_12.append(kp1[match[0].queryIdx].pt)
            kp2_match_12.append(kp2[match[0].trainIdx].pt)
    elif len(match)==1:
        matches12.append(match[0])
        kp1_match_12.append(kp1[match[0].queryIdx].pt)
        kp2_match_12.append(kp2[match[0].trainIdx].pt)

kp1_match_12 = np.ascontiguousarray(kp1_match_12)
kp2_match_12 = np.ascontiguousarray(kp2_match_12)
        
kp1_match_12_ud = cv2.undistortPoints(np.expand_dims(kp1_match_12,axis=1),K,D)
kp2_match_12_ud = cv2.undistortPoints(np.expand_dims(kp2_match_12,axis=1),K,D)

#print("kp1",kp1[0].pt,dst1[0])

E_12, mask_e_12 = cv2.findEssentialMat(kp1_match_12_ud, kp2_match_12_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e_12) ," of total ",len(matches12),"matches")

points, R_12, t_12, mask_RP_12 = cv2.recoverPose(E_12, kp1_match_12_ud, kp2_match_12_ud,mask=mask_e_12)
print("points:",points,"\trecover pose mask:",np.sum(mask_RP_12!=0))
print("R:",R_12)
print("t:",t_12.T)

img12 = displayMatches(gr1,kp1,gr2,kp2,matches12,mask_RP_12, False)
fig1 = plt.figure(1)
plt.imshow(img12),
plt.ion()
#plt.show()
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")



'''
Pose_1 = np.dot(K,np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
print ("Pose_1: ", Pose_1)
Pose_2 = np.dot(K, np.hstack((R_12, t_12)))
print ("Pose_2: ", Pose_2)
'''

#Create 3 x 4 Homogenous Transform
Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
print ("Pose_1: ", Pose_1)
Pose_2 = np.hstack((R_12, t_12))
print ("Pose_2: ", Pose_2)

#P_l = np.dot(K,  M_l)
#P_r = np.dot(K,  M_r)
#print("dst: ",dst1)

# Points Given in N,1,2 array 
landmarks_12_hom = cv2.triangulatePoints(Pose_1, Pose_2, 
                                     kp1_match_12_ud[mask_RP_12[:,0]==1], 
                                     kp2_match_12_ud[mask_RP_12[:,0]==1]).T
landmarks_12_hom_norm = landmarks_12_hom /  landmarks_12_hom[:,-1][:,None]
landmarks_12 = landmarks_12_hom_norm[:, :3]


fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
#graph, = ax.plot(landmarks_12[:,0], landmarks_12[:,1], landmarks_12[:,2], linestyle="", marker="o")
graph = plot_3d_points(ax, landmarks_12, linestyle="", marker="o")

if CHESSBOARD:
    ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
    ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)
    ret3, corners3 = cv2.findChessboardCorners(gr3, (16,9),None)
    
    corners1_ud = cv2.undistortPoints(corners1,K,D)
    corners2_ud = cv2.undistortPoints(corners2,K,D)
    corners3_ud = cv2.undistortPoints(corners3,K,D)
    
    corners_12_hom = cv2.triangulatePoints(Pose_1, Pose_2, corners1_ud, corners2_ud).T
    corners_12_hom_norm = corners_12_hom /  corners_12_hom[:,-1][:,None]
    corners_12 = corners_12_hom_norm[:, :3]
    #print(point_3d)
    
    graph = plot_3d_points(ax, corners_12, linestyle="", marker=".",color='g')

#plot_pose3_on_axes(ax,np.linalg.inv(R_12),-t_12.T, axis_length=1.0)
#R_12_inv, t_12_inv = pose_inv(R_12, t_12)

plot_pose3_on_axes(ax, R_12.T, -t_12.T, axis_length=1.0)
plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=0.5)

set_axes_equal(ax)
ax.view_init(-70, -90)
#ax.set_zlim3d(-2,5)
plt.show()
#plt.ion()
#plt.draw()
#plt.pause(2.0)

# important!
#plt.show()

'''
process frame
where the state of a frame at time t , , contains the following data:
S: A set of 2D keypoints {p } (each one of them is associated to a 3D landmark
k k=1..K
The associated set of 3D landmarks {X } .i
'''
matches23 = []
kp2_match_23 = []
kp3_match_23 = []
for i,match in enumerate(matches23_knn):
    if len(match)>1:
        if match[0].distance < 0.80*match[1].distance:
            matches23.append(match[0])
            kp2_match_23.append(kp2[match[0].queryIdx].pt)
            kp3_match_23.append(kp3[match[0].trainIdx].pt)
    elif len(match)==1:
        matches23.append(match[0])
        kp2_match_23.append(kp2[match[0].queryIdx].pt)
        kp3_match_23.append(kp3[match[0].trainIdx].pt)

kp2_match_23 = np.ascontiguousarray(kp2_match_23)
kp3_match_23 = np.ascontiguousarray(kp3_match_23)

lm = np.zeros(mask_RP_12.shape[0],dtype=int)
lm[mask_RP_12.ravel()==1]=np.arange(np.sum(mask_RP_12))

frame2_to_lm = {mat.trainIdx:lm_id for lm_id,mat in zip(lm, matches12)
                if lm_id!=0 }

# Filter frame 2 to frame 3 matches
kp2_match_23 = np.array([kp2[mat.queryIdx].pt for mat in matches23])
kp3_match_23 = np.array([kp3[mat.trainIdx].pt for mat in matches23])

kp2_match_23_ud = undistortKeyPoints(kp2_match_23,K,D)
kp3_match_23_ud = undistortKeyPoints(kp3_match_23,K,D)

E_23, mask_e_23 = cv2.findEssentialMat(kp2_match_23_ud, kp3_match_23_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e_23) ," of total ",len(matches23),"matches")

points, R_23, t_23, mask_RP_23 = cv2.recoverPose(E_23, kp2_match_23_ud, kp3_match_23_ud,mask=mask_e_23)
img23 = displayMatches(gr2,kp2,gr3,kp3,matches23,mask_RP_23, False)

fig1 = plt.figure(1)
plt.imshow(img23)
#plt.ion()
#plt.show()
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

#img23 = displayMatches(gr2,kp2,gr3,kp3,matches23,mask_RP_23, False)
#time.sleep(3)
#plt.imshow(img23),plt.show()

#matches23_filt = [matches23[i] for i in range(len(matches23)) if mask_RP_23[i]==1]
matches23_filt = matches23
#frame 3 stuff
frame3_to_frame2 = {mat.trainIdx:mat.queryIdx for mat in matches23_filt}

frame3_to_lm = {id:frame2_to_lm.get(frame3_to_frame2[id]) 
                for id in frame3_to_frame2.keys() 
                if frame2_to_lm.get(frame3_to_frame2[id]) is not None}
print("Frame3_to_lm: ",len(frame3_to_lm))

landmarks_23 = np.array([landmarks_12[frame3_to_lm[k]] for k in 
                        frame3_to_lm.keys()])

lm_kps_3 = np.array([kp3[k].pt for k in frame3_to_lm.keys()])

(success, rvec_23, t_23_inv, inliers) = cv2.solvePnPRansac(landmarks_23, lm_kps_3, K, D)# flags=cv2.SOLVEPNP_ITERATIVE
R_23_inv, jacobian	=	cv2.Rodrigues(rvec_23)
t_23 = np.matmul((-R_23_inv).T,t_23_inv).T   
R_23 = R_23_inv.T


plt.figure(2)
graph = plot_3d_points(ax, landmarks_23, linestyle="", marker="o", color='r')

plot_pose3_on_axes(ax,R_23, t_23, axis_length=5.0)

Pose_3 = np.hstack((R_23, t_12))
print ("Pose_3: ", Pose_3)

if CHESSBOARD:
    corners_23_hom = cv2.triangulatePoints(Pose_2, Pose_3, corners2_ud, corners3_ud).T
    corners_23_hom_norm = corners_23_hom /  corners_23_hom[:,-1][:,None]
    corners_23 = corners_23_hom_norm[:, :3]
    
    graph = plot_3d_points(ax, corners_23, linestyle="", marker=".",color='tab:orange')

set_axes_equal(ax)             # important!
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")