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

kp1_matched_pts = [kp1[mat.queryIdx] for mat in matches] 
kp2_matched_pts = [kp2[mat.trainIdx] for mat in matches]

dst1 = undistortKeyPoints(kp1_matched_pts,K,D)
dst2 = undistortKeyPoints(kp2_matched_pts,K,D)

print("kp1",kp1[0].pt,dst1[0])

E, mask = cv2.findEssentialMat(dst1, dst2, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)
print ("Essential matrix used ",np.asscalar(sum(mask)) ," of total ",len(matches),"matches")

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
print("dst: ",dst1)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, 
                                     dst1[mask_recPose[:,0]==1].T, 
                                     dst2[mask_recPose[:,0]==1].T).T
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
plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=1.0)

set_axes_equal(ax)             # important!

plt.show()
