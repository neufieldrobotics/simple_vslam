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
else:    
    img1 = cv2.imread('/home/vik748/data/lab_timelapse2/G0050894.JPG',1)          # queryImage
    img2 = cv2.imread('/home/vik748/data/lab_timelapse2/G0050899.JPG',1)  

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

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

# create BFMatcher object - Brute Force
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

#matches = sorted(matches, key = lambda x:x.distance)

kp1_matched_pts = np.array([kp1[mat.queryIdx].pt for mat in matches])
kp2_matched_pts = np.array([kp2[mat.trainIdx].pt for mat in matches])

dst1 = undistortKeyPoints(kp1_matched_pts,K,D)
dst2 = undistortKeyPoints(kp2_matched_pts,K,D)

print("kp1",kp1[0].pt,dst1[0])

E, mask = cv2.findEssentialMat(dst1, dst2, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix used ",np.sum(mask) ," of total ",len(matches),"matches")

img5 = displayMatches(gr1,kp1,gr2,kp2,matches,mask)

points, R, t, mask_recPose = cv2.recoverPose(E, dst1, dst2,mask=mask)
print("points:",points)
print("R:",R)
print("t:",t.transpose())
print("recover pose mask:",np.sum(mask_recPose!=0))

plt.imshow(img5),plt.show()

M_r = np.hstack((R, t))
print ("M_r: ", M_r)
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
print ("M_l: ", M_l)

P_l = np.dot(K,  M_l)
P_r = np.dot(K,  M_r)
#print("dst: ",dst1)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, 
                                     kp1_matched_pts[mask_recPose[:,0]==1].T, 
                                     kp2_matched_pts[mask_recPose[:,0]==1].T).T
#point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_4d = point_4d_hom /  point_4d_hom[:,-1][:,None]
point_3d = point_4d[:, :3]
#print(point_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
graph, = ax.plot(point_3d[:,0], point_3d[:,1], point_3d[:,2], linestyle="", marker="o")

plot_pose3_on_axes(ax,R,t.T, axis_length=1.0)
plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=2.0)

#set_axes_equal(ax)             # important!

#plt.show()

'''
process frame
where the state of a frame at time t , , contains the following data:
S: A set of 2D keypoints {p } (each one of them is associated to a 3D landmark
k k=1..K
The associated set of 3D landmarks {X } .i
'''
class Frames:    
    seq = 0   
    def __init__(self, gs_image, orb):
        #self.img = gs_image
        self.keypoints, self.des = orb.detectAndCompute(gs_image,None)
        self.keypoint_match = np.full(len(self.keypoints), False)
        self.pose_t = np.zeros(3)
        self.pose_R = np.zeros((3,3))
        self.id = self.__class__.seq 
        self.__class__.seq  += 1
    def match_prev_frame(self, prev_frame, matcher):
        #self.img = gs_image
        self.matches = matcher.match(self.des,prev_frame.des)
        self.match_dict = {mat.queryIdx:mat.trainIdx for mat in self.matches}
        # lm[mask_recPose.ravel()==1]=np.arange(np.sum(mask_recPose))
        # {mat.queryIdx:lm for lm,mat in zip(lm.astype(int), matches) if lm!=0}
 #   def matched_keypoints:
 #       return self.keypoints
        
frames_list = []
if sys.platform == 'darwin':
    curr_frame = cv2.imread('/Users/vik748/data/lab_timelapse2/G0050900.JPG',1)          # queryImage
else:    
    curr_frame = cv2.imread('/home/vik748/data/lab_timelapse2/G0050900.JPG',1)          # queryImage

second_frame = Frames(gr2,orb)
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
