#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from itertools import compress



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

def undistortKeyPoints(kps, K, D):
  '''
  This function extracts coordinates from keypoint object,
  undistorts them using K and D and returns undistorted coordinates"
  '''
  kp_pts = np.array([o.pt for o in kps])
  kp_pts_cont = np.ascontiguousarray(kp_pts[:,:2]).reshape((kp_pts.shape[0],1,2))
  # this version returns normalized points with F=1 and centered at 0,0
  # cv2.undistortPoints(kp_pts_cont, K, D,  noArray(), K) would return unnormalized output
  return	cv2.undistortPoints(kp_pts_cont, K, D)

H, W = img1.shape[:2]
Kprime, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0, (W, H))
x, y, w, h = roi

print(Kprime,roi)

#mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, Kprime, (W,H), 5)

#undist_1 = cv2.remap(gr1, mapx, mapy, cv2.INTER_LINEAR)
#undist_2 = cv2.remap(gr2, mapx, mapy, cv2.INTER_LINEAR)

#x, y, w, h = roi
#undist_1 = undist_1[y:y + h, x:x + w]
#undist_2 = undist_2[y:y + h, x:x + w]

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gr1,None)
kp2, des2 = orb.detectAndCompute(gr2,None)

# create BFMatcher object - Brute Force
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

print("Size of matches: ", len(matches))

#matches = sorted(matches, key = lambda x:x.distance)

kp1_matched_pts = [kp1[mat.queryIdx] for mat in matches] 
kp2_matched_pts = [kp2[mat.trainIdx] for mat in matches]

dst1 = undistortKeyPoints(kp1_matched_pts,K,D)
dst2 = undistortKeyPoints(kp2_matched_pts,K,D)

print("kp1",kp1[0].pt,dst1[0])

E, mask = cv2.findEssentialMat(dst1, dst2, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print type(mask.astype(bool))
bool_mask = mask.astype(bool)
print(bool_mask)
validmatches = list(compress(matches, bool_mask.ravel().tolist()))
print ("len of validmatches ", len(validmatches),"len of matches ", len(matches))

img4 = cv2.drawMatches(gr1,kp1,gr2,kp2,matches, None, matchColor=(0, 255, 0), 
                       matchesMask=bool_mask.ravel().tolist(), flags=2)

img5 = cv2.drawMatches(gr1,kp1,gr2,kp2,matches, img4, matchColor=(255, 0, 0), 
                       matchesMask=np.invert(bool_mask).ravel().tolist(), flags=1)

                              

print("compute E mask:",np.sum(mask),"of",len(mask))

points, R, t, mask = cv2.recoverPose(E, dst1, dst2)
print("points:",points)
print("R:",R)
print("t:",t)
print("recover pose mask:",np.sum(mask))

plt.imshow(img5),plt.show()