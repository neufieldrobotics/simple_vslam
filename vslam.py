# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


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

H, W = img1.shape[:2]
Kprime, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0, (W, H))
x, y, w, h = roi

print(Kprime,roi)

mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, Kprime, (W,H), 5)

undist_1 = cv2.remap(gr1, mapx, mapy, cv2.INTER_LINEAR)
undist_2 = cv2.remap(gr2, mapx, mapy, cv2.INTER_LINEAR)

x, y, w, h = roi
undist_1 = undist_1[y:y + h, x:x + w]
undist_2 = undist_2[y:y + h, x:x + w]

#plt.imshow(gr1),plt.show()
#plt.imshow(undist_1),plt.show()


#cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
#cv2.resizeWindow("output", 800, 600)              # Resize window to specified dimensions


        # queryImage
#img1 = cv2.imread('simple.jpg',0)

#print (img1)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(undist_1,None)
kp2, des2 = orb.detectAndCompute(undist_2,None)

print("kp1",kp1[1].pt,kp1[1].angle)
print("des1",des1[1])

#img1 = cv2.drawKeypoints(img1,kp,color=(0,255,0), flags=0)
#cv2.imshow("output", img1)                            # Show image
#cv2.waitKey(1000)
#cv2.imshow("output", gr1)                            # Show image
#cv2.waitKey(1000)
#plt.show()

img3 = cv2.drawKeypoints(img1, kp1, None,color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
#plt.imshow(img3), plt.show()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#bf = cv2.BFMatcher()


# Match descriptors.
matches = bf.match(des1,des2)
#matches = bf.knnMatch(des1,des2, k=2)
print(matches[1])

# Apply ratio test
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        good.append([m,n])
#print("Size of good: ", len(good))
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
 
# Draw first 10 matches.
print("Size of matches: ", len(matches))
img4 = cv2.drawMatches(undist_1,kp1,undist_1,kp2,matches[0:100], None,flags=2)
 
plt.imshow(img4),plt.show()
