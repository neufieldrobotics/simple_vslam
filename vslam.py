#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from vslam_helper import *
 
print (sys.platform)

# Inputs, images and camera info
if sys.platform == 'darwin':
    img1 = cv2.imread('/Users/vik748/data/lab_timelapse2/G0050894.JPG',1)          # queryImage
    img2 = cv2.imread('/Users/vik748/data/lab_timelapse2/G0050899.JPG',1)  
    img3 = cv2.imread('/Users/vik748/data/lab_timelapse2/G0050899.JPG',1)  
else:    
    img1 = cv2.imread('/home/vik748/data/test_set/GOPR1429.JPG',1)          # queryImage
    img2 = cv2.imread('/home/vik748/data/test_set/GOPR1430.JPG',1)  
    img3 = cv2.imread('/home/vik748/data/test_set/GOPR1431.JPG',1)  

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gr3=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

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
orb = cv2.ORB_create()

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

Pose_1 = np.dot(K,np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
print ("Pose_1: ", Pose_1)
Pose_2 = np.dot(K, np.hstack((R_12, t_12)))
print ("Pose_2: ", Pose_2)

#P_l = np.dot(K,  M_l)
#P_r = np.dot(K,  M_r)
#print("dst: ",dst1)
landmarks_12_hom = cv2.triangulatePoints(Pose_1, Pose_2, 
                                     kp1_match_12[mask_RP_12[:,0]==1].T, 
                                     kp2_match_12[mask_RP_12[:,0]==1].T).T
#point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
landmarks_12_hom_norm = landmarks_12_hom /  landmarks_12_hom[:,-1][:,None]
landmarks_12 = landmarks_12_hom_norm[:, :3]
#print(point_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
graph, = ax.plot(point_3d[:,0], point_3d[:,1], point_3d[:,2], linestyle="", marker="o")

plot_pose3_on_axes(ax,R_12,t_12.T, axis_length=1.0)
plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=2.0)

set_axes_equal(ax)             # important!

plt.show()

'''
process frame
where the state of a frame at time t , , contains the following data:
S: A set of 2D keypoints {p } (each one of them is associated to a 3D landmark
k k=1..K
The associated set of 3D landmarks {X } .i
'''

lm = np.zeros(mask_recPose.shape[0],dtype=int)
lm[mask_recPose.ravel()==1]=np.arange(np.sum(mask_recPose))

second_frame.lm_matches = {mat.trainIdx:lm_id for lm_id,mat in zip(lm, matches)
                           if lm_id!=0 }

curr_frame_gr=cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY)
frames_list.append(Frames(curr_frame_gr,orb))
frames_list[0].match_prev_frame(second_frame,bf)
frames_list[0].lm_matches = {id:second_frame.lm_matches.get(frames_list[0].match_dict[id]) 
                             for id in frames_list[0].match_dict.keys() 
                             if second_frame.lm_matches.get(frames_list[0].match_dict[id]) 
                             is not None}
print(len(frames_list[0].lm_matches))
lm3d = np.array([point_3d[frames_list[0].lm_matches[k]] for k in 
                 frames_list[0].lm_matches.keys()])
pts2d = np.array([frames_list[0].keypoints[k].pt for k in frames_list[0].lm_matches.keys()])

(success, rvec, t) = cv2.solvePnP(lm3d, pts2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
R, jacobian	=	cv2.Rodrigues(rvec)


plot_pose3_on_axes(ax,R,t.T, axis_length=5.0)

set_axes_equal(ax)             # important!

plt.show()
