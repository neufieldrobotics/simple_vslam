#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from vslam_helper import *
 
print (sys.platform)

# Inputs, images and camera info
if sys.platform == 'darwin':
    img1 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1429.JPG',1)          # queryImage
    img2 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1430.JPG',1)  
    img3 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1431.JPG',1)  
else:    
    img1 = cv2.imread('/home/vik748/data/test_set/GOPR1429.JPG',1)          # queryImage
    img2 = cv2.imread('/home/vik748/data/test_set/GOPR1430.JPG',1)  
    img3 = cv2.imread('/home/vik748/data/test_set/GOPR1431.JPG',1)  

# Inputs, images and camera info
if sys.platform == 'darwin':
    img1 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1429.JPG',1)          # queryImage
    img2 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1430.JPG',1)  
    img3 = cv2.imread('/Users/vik748/Google Drive/data/test_set/GOPR1431.JPG',1)  
else:    
    img1 = cv2.imread('/home/vik748/data/chess_board/GOPR1453.JPG',1)          # queryImage
    img2 = cv2.imread('/home/vik748/data/chess_board/GOPR1452.JPG',1)  
    img3 = cv2.imread('/home/vik748/data/chess_board/GOPR1451.JPG',1)  

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gr3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])

D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);

print(K,D)


# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=25000)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gr1,None)
kp2, des2 = orb.detectAndCompute(gr2,None)
kp3, des3 = orb.detectAndCompute(gr3,None)

# create BFMatcher object - Brute Force
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches12 = bf.match(des1,des2)
matches23 = bf.match(des2,des3)

#matches = sorted(matches, key = lambda x:x.distance)

kp1_match_12 = np.array([kp1[mat.queryIdx].pt for mat in matches12])
kp2_match_12 = np.array([kp2[mat.trainIdx].pt for mat in matches12])

kp1_match_12_ud = undistortKeyPoints(kp1_match_12,K,D)
kp2_match_12_ud = undistortKeyPoints(kp2_match_12,K,D)

#print("kp1",kp1[0].pt,dst1[0])

E_12, mask_e_12 = cv2.findEssentialMat(kp1_match_12_ud, kp2_match_12_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.01)

print ("Essential matrix: used ",np.sum(mask_e_12) ," of total ",len(matches12),"matches")

points, R_12, t_12, mask_RP_12 = cv2.recoverPose(E_12, kp1_match_12_ud, kp2_match_12_ud,mask=mask_e_12)
print("points:",points,"\trecover pose mask:",np.sum(mask_RP_12!=0))
print("R:",R_12)
print("t:",t_12.transpose())


img12 = displayMatches(gr1,kp1,gr2,kp2,matches12,mask_RP_12)
plt.imshow(img12),plt.show()

'''
Pose_1 = np.dot(K,np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
print ("Pose_1: ", Pose_1)
Pose_2 = np.dot(K, np.hstack((R_12, t_12)))
print ("Pose_2: ", Pose_2)
'''

Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
print ("Pose_1: ", Pose_1)
Pose_2 = np.hstack((R_12, t_12))
print ("Pose_2: ", Pose_2)

#P_l = np.dot(K,  M_l)
#P_r = np.dot(K,  M_r)
#print("dst: ",dst1)
landmarks_12_hom = cv2.triangulatePoints(Pose_1, Pose_2, 
                                     kp1_match_12_ud[mask_RP_12[:,0]==1].T, 
                                     kp2_match_12_ud[mask_RP_12[:,0]==1].T).T
#point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
landmarks_12_hom_norm = landmarks_12_hom /  landmarks_12_hom[:,-1][:,None]
landmarks_12 = landmarks_12_hom_norm[:, :3]
#print(point_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
graph, = ax.plot(landmarks_12[:,0], landmarks_12[:,1], landmarks_12[:,2], linestyle="", marker="o")

plot_pose3_on_axes(ax,np.linalg.inv(R_12),-t_12.T, axis_length=1.0)
plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=1.0)

set_axes_equal(ax)             # important!

plt.show()

'''
process frame
where the state of a frame at time t , , contains the following data:
S: A set of 2D keypoints {p } (each one of them is associated to a 3D landmark
k k=1..K
The associated set of 3D landmarks {X } .i
'''

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
                               method=cv2.RANSAC, prob=0.999, threshold=0.01)

print ("Essential matrix: used ",np.sum(mask_e_23) ," of total ",len(matches23),"matches")

points, R_23, t_23, mask_RP_23 = cv2.recoverPose(E_23, kp2_match_23_ud, kp3_match_23_ud,mask=mask_e_23)

#img23 = displayMatches(gr2,kp2,gr3,kp3,matches23,mask_RP_23)
#time.sleep(3)
#plt.imshow(img23),plt.show()

matches23_filt = [matches23[i] for i in range(len(matches23)) if mask_RP_23[i]==1]

#frame 3 stuff
frame3_to_frame2 = {mat.trainIdx:mat.queryIdx for mat in matches23_filt}

frame3_to_lm = {id:frame2_to_lm.get(frame3_to_frame2[id]) 
                for id in frame3_to_frame2.keys() 
                if frame2_to_lm.get(frame3_to_frame2[id]) is not None}
print(len(frame3_to_lm))

landmarks_23 = np.array([landmarks_12[frame3_to_lm[k]] for k in 
                        frame3_to_lm.keys()])

lm_kps_3 = np.array([kp3[k].pt for k in frame3_to_lm.keys()])

(success, rvec_23, t_23) = cv2.solvePnP(landmarks_23, lm_kps_3, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
R_23, jacobian	=	cv2.Rodrigues(rvec_23)
camera_position = np.matmul((-R_23).T,t_23)    


graph, = ax.plot(landmarks_23[:,0], landmarks_23[:,1], landmarks_23[:,2], linestyle="", marker="o", color='r')

plot_pose3_on_axes(ax,R_23.T,camera_position.T, axis_length=5.0)

set_axes_equal(ax)             # important!
plt.show()
