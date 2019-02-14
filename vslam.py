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

RADIAL_NON_MAX = True
CHESSBOARD = True

# Inputs, images and camera info


if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
    window_xadj = 0
    window_yadj = 45
else:
    path = '/home/vik748/'
    window_xadj = 65
    window_yadj = 430
img1 = cv2.imread(path+'data/chess_board2/GOPR1492.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board2/GOPR1493.JPG',1)  
img3 = cv2.imread(path+'data/chess_board2/GOPR1494.JPG',1)  
img4 = cv2.imread(path+'data/chess_board2/GOPR1497.JPG',1)

mask_pts_1488 = np.array([[1180, 960], [2740, 1000], [2700, 2040], [1180, 1980]])
mask_pts_1489 = np.array([[1180, 1225], [2550, 1475], [2400, 2340], [1100, 2100]])
mask_pts_1490 = np.array([[1550, 1250], [2870, 1030], [2900, 1890], [1680, 2200]])
mask_pts_1491 = np.array([[760, 880], [2100, 1090], [2180, 1980], [780, 2100]])
mask_pts_1492 = np.array([[1697, 1227], [2600, 1271], [2560, 1840], [1670, 1817]])
mask_pts_1493 = np.array([[1346, 1021], [2231, 1043], [2210, 1643], [1330, 1595]])
mask_pts_1494 = np.array([[1102, 1000], [1981, 1027], [1965, 1622], [1102, 1595]])
mask_pts_1496 = np.array([[400, 650], [1300, 680], [1280, 1300], [390, 1260]])
mask_pts_1497 = np.array([[760, 1055], [1600, 1080], [1560, 1670], [755, 1630]])
mask_pts_1498 = np.array([[1030, 740], [1820, 840], [1785, 1085], [1675, 1400], [975, 1285]])

mask = np.zeros(img1.shape[:2], dtype=np.uint8)

mask1 = 255 - cv2.fillConvexPoly(mask, mask_pts_1492, color=[255, 255, 255])
mask2 = 255 - cv2.fillConvexPoly(mask, mask_pts_1493, color=[255, 255, 255])
mask3 = 255 - cv2.fillConvexPoly(mask, mask_pts_1494, color=[255, 255, 255])
mask4 = 255 - cv2.fillConvexPoly(mask, mask_pts_1497, color=[255, 255, 255])
'''
img1 = cv2.imread(path+'data/chess_board2/GOPR1496.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board2/GOPR1497.JPG',1)  
img3 = cv2.imread(path+'data/chess_board2/GOPR1498.JPG',1)
'''

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

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
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
'''
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 1
matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 20,
                                   multi_probe_level = 2), dict(checks=100))


detector = cv2.xfeatures2d.SIFT_create(edgeThreshold = 7, nOctaveLayers = 3)
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 1
Smatcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, table_number = 6, key_size = 20,
                                   multi_probe_level = 2), dict(checks=100))
# Match descriptors.
'''                     
#detector = cv2.AKAZE_create(threshold=.0005)
#detector = cv2.BRISK_create(thresh = 15, octaves = 10, patternScale = 1.0 )
# find the keypoints and descriptors with ORB
kp1 = detector.detect(gr1,mask1)
kp2 = detector.detect(gr2,mask2)

print ("Points detected: ",len(kp1))

'''
kp1 = sorted(kp1, key = lambda x:x.response, reverse = True)
kp2 = sorted(kp2, key = lambda x:x.response, reverse = True)
kp3 = sorted(kp3, key = lambda x:x.response, reverse = True)
print ("Points sorted: ")

kp1 = SSC(kp1, 5000, 0.1, gr1.shape[1], gr1.shape[0])
kp2 = SSC(kp2, 5000, 0.1, gr1.shape[1], gr1.shape[0])
kp3 = SSC(kp3, 5000, 0.1, gr1.shape[1], gr1.shape[0])
print ("Points nonmax supression: ")
'''
if RADIAL_NON_MAX:
    kp1 = radial_non_max(kp1,25)
    kp2 = radial_non_max(kp2,25)
    print ("Points after radial supression: ",len(kp1))


kp1, des1 = detector.compute(gr1,kp1)
kp2, des2 = detector.compute(gr2,kp2)
print ("Descriptors computed: ")

# create BFMatcher object - Brute Force
'''
kp1_match_12, kp2_match_12, matches12 = knn_match_and_filter(matcher, kp1, kp2, des1, des2)
kp2_match_23, kp3_match_23, matches23 = knn_match_and_filter(matcher, kp2, kp3, des2, des3)

'''
matches12 = matcher.match(des1,des2)

kp1_match_12 = np.array([kp1[mat.queryIdx].pt for mat in matches12])
kp2_match_12 = np.array([kp2[mat.trainIdx].pt for mat in matches12])

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
plt.get_current_fig_manager().window.setGeometry(window_xadj,window_yadj,640,338) #(0, 0, 800, 900)
#move_figure(position="left")
plt.imshow(img12)
plt.title('Image 1 to 2 matches')
#plt.ion()
#plt.show()
plt.axis("off")
fig1.subplots_adjust(0,0,1,1)
plt.draw()
plt.pause(0.001)

fig3 = plt.figure(3)
plt.get_current_fig_manager().window.setGeometry(window_xadj,338+window_yadj,640,338) #(0, 0, 800, 900)
img2_track = draw_feature_tracks(gr1,kp1,gr2,kp2,matches12,mask_RP_12)
plt.imshow(img2_track)
plt.title('Image 1 to 2 matches')
#plt.ion()
#plt.show()
plt.axis("off")
fig3.subplots_adjust(0,0,1,1)
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

landmarks_12 = triangulate(np.eye(4), T_1_2, kp1_match_12_ud[mask_RP_12[:,0]==1], 
                                             kp2_match_12_ud[mask_RP_12[:,0]==1])

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
#move_figure(position="right")
ax2.set_aspect('equal')         # important!
title = ax2.set_title('Image 1 to 2 after triangulation')
graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker="o")

if CHESSBOARD:
    ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
    ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)
    
    corners1_ud = cv2.undistortPoints(corners1,K,D)
    corners2_ud = cv2.undistortPoints(corners2,K,D)
   
    corners_12 = triangulate(np.eye(4), T_1_2, corners1_ud, corners2_ud)
    graph = plot_3d_points(ax2, corners_12, linestyle="", marker=".",color='g')

plot_pose3_on_axes(ax2, T_1_2, axis_length=1.0)
plot_pose3_on_axes(ax2,np.eye(4), axis_length=0.5)

set_axes_equal(ax2)
ax2.view_init(-70, -90)
fig1.subplots_adjust(0,0,1,1)

plt.draw()
plt.pause(.001)
input("Press [enter] to continue.")

lm_12 = -np.ones(mask_RP_12.shape[0],dtype=int)
lm_12[mask_RP_12.ravel()==1]=np.arange(np.sum(mask_RP_12))

# Create a dictionary {KP2 index of match : landmark number}
frame2_to_lm = {mat.trainIdx:lm_id for lm_id,mat in zip(lm_12, matches12)
                if lm_id!=-1 }
lm_to_frame2 = dict([[v,k] for k,v in frame2_to_lm.items()])
frame2_to_matches12 = {mat.trainIdx:match_id for match_id,mat in enumerate(matches12)}

'''
process frame
where the state of a frame at time t , , contains the following data:
S: A set of 2D keypoints {p } (each one of them is associated to a 3D landmark
k k=1..K
The associated set of 3D landmarks {X } .i
'''
gr3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
kp3 = detector.detect(gr3,mask3)

if RADIAL_NON_MAX:
    kp3 = radial_non_max(kp3,25)
    print ("Points after radial supression: ",len(kp1))

kp3, des3 = detector.compute(gr3,kp3)

matches23 = matcher.match(des2,des3)
kp2_match_23 = np.array([kp2[mat.queryIdx].pt for mat in matches23])
kp3_match_23 = np.array([kp3[mat.trainIdx].pt for mat in matches23])

kp2_match_23_ud = cv2.undistortPoints(np.expand_dims(kp2_match_23,axis=1),K,D)
kp3_match_23_ud = cv2.undistortPoints(np.expand_dims(kp3_match_23,axis=1),K,D)

E_23, mask_e_23 = cv2.findEssentialMat(kp2_match_23_ud, kp3_match_23_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e_23) ," of total ",len(matches23),"matches")
#mask_RP_23 = mask_e_23
points, R_32, t_32, mask_RP_23 = cv2.recoverPose(E_23, kp2_match_23_ud, kp3_match_23_ud,mask=mask_e_23)

matches23_filt = [matches23[i] for i in range(len(matches23)) if mask_RP_23[i]==1]

frame3_to_frame2 = {mat.trainIdx:mat.queryIdx for mat in matches23_filt}

frame3_to_lm = {id:frame2_to_lm.get(frame3_to_frame2[id]) 
                for id in frame3_to_frame2.keys() 
                if frame2_to_lm.get(frame3_to_frame2[id]) is not None}

fig1 = plt.figure(1)
plt.title('Image 1 to 2 - Landmarks found in 3')
mask_lm3_in_12 = np.zeros(mask_RP_12.shape)

for frame3_kp, lm_id in frame3_to_lm.items():
    #print (lm_id)
    frame2_kp = lm_to_frame2[lm_id]
    matches_12_id = frame2_to_matches12[frame2_kp]
    mask_lm3_in_12[matches_12_id]=1.0
    
img12_lm = displayMatches(gr1,kp1,gr2,kp2,matches12,mask_lm3_in_12, False, in_image=img12, color=(255,165,0))
plt.imshow(img12_lm)
plt.draw()
plt.pause(.001)
input("Press [enter] to continue.")


img23 = displayMatches(gr2,kp2,gr3,kp3,matches23,mask_RP_23, False)

plt.imshow(img23)
plt.title('Image 2 to 3')

fig3 = plt.figure(3)
img3_track = draw_feature_tracks(gr2,kp2,gr3,kp3,matches23,mask_RP_23)
plt.imshow(img3_track)
plt.title('Image 2 to 3 matches')
plt.draw()
plt.pause(0.001)

input("Press [enter] to continue.")

print("Frame3_to_lm: ",len(frame3_to_lm))

landmarks_23 = np.array([landmarks_12[frame3_to_lm[k]] for k in 
                        frame3_to_lm.keys()])

lm_kps_3 = np.array([kp3[k].pt for k in frame3_to_lm.keys()])
success, T_2_3, inliers = T_from_PNP(landmarks_23, lm_kps_3, K, D)

plt.figure(2)
plt.title('Image 2 to 3 PNP')
graph = plot_3d_points(ax2, landmarks_23, linestyle="", marker="o", color='r')
plot_pose3_on_axes(ax2, T_2_3, axis_length=2.0)

if CHESSBOARD:
    ret3, corners3 = cv2.findChessboardCorners(gr3, (16,9),None)
    corners3_ud = cv2.undistortPoints(corners3,K,D)

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
plt.title('Image 2 to 3 New Landmarks')
plt.draw()
plt.pause(0.01)
input("Press [enter] to continue.")

lm_23 = -np.ones(mask_RP_23.shape[0],dtype=int)
lm_23[mask_RP_23.ravel()==1]=np.arange(np.sum(mask_RP_23))

# Create a dictionary {KP2 index of match : landmark number}
frame3_to_lm = {mat.trainIdx:lm_id for lm_id,mat in zip(lm_23, matches23)
                if lm_id!=-1 }
lm_to_frame3 = dict([[v,k] for k,v in frame3_to_lm.items()])
frame3_to_matches23 = {mat.trainIdx:match_id for match_id,mat in enumerate(matches23)}

'''
FRAME 4
'''
def process_frame(img_curr, mask_curr):
    gr_curr=cv2.cvtColor(img_curr,cv2.COLOR_BGR2GRAY)
    kp_curr = detector.detect(gr_curr,mask_curr)
    
    if RADIAL_NON_MAX:
        kp_curr = radial_non_max(kp_curr,25)
        print ("Points after radial supression: ",len(kp_curr))
    
    kp_curr, des_curr = detector.compute(gr_curr,kp_curr)
    
    matches34 = matcher.match(des3,des_curr)
    kp3_match_34 = np.array([kp3[mat.queryIdx].pt for mat in matches34])
    kp4_match_34 = np.array([kp_curr[mat.trainIdx].pt for mat in matches34])
    
    kp3_match_34_ud = cv2.undistortPoints(np.expand_dims(kp3_match_34,axis=1),K,D)
    kp4_match_34_ud = cv2.undistortPoints(np.expand_dims(kp4_match_34,axis=1),K,D)
    
    E_34, mask_e_34 = cv2.findEssentialMat(kp3_match_34_ud, kp4_match_34_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, prob=0.999, threshold=0.001)
    
    print ("Essential matrix: used ",np.sum(mask_e_34) ," of total ",len(matches34),"matches")
    #mask_RP_23 = mask_e_23
    points, R_32, t_32, mask_RP_34 = cv2.recoverPose(E_34, kp3_match_34_ud, kp4_match_34_ud,mask=mask_e_34)
    
    matches34_filt = [matches34[i] for i in range(len(matches34)) if mask_RP_34[i]==1]
    
    frame4_to_frame3 = {mat.trainIdx:mat.queryIdx for mat in matches34_filt}
    
    frame4_to_lm = {id:frame3_to_lm.get(frame4_to_frame3[id]) 
                    for id in frame4_to_frame3.keys() 
                    if frame3_to_lm.get(frame4_to_frame3[id]) is not None}
    
    fig1 = plt.figure(1)
    plt.title('Image 1 to 2 - Landmarks found in 3')
    mask_lm4_in_23 = np.zeros(mask_RP_23.shape)
    
    for frame4_kp, lm_id in frame4_to_lm.items():
        #print (lm_id)
        frame3_kp = lm_to_frame3[lm_id]
        matches_23_id = frame3_to_matches23[frame3_kp]
        mask_lm4_in_23[matches_23_id]=1.0
        
    img23_lm = displayMatches(gr2,kp2,gr3,kp3,matches23,mask_lm4_in_23, False, in_image=img23, color=(255,165,0))
    plt.imshow(img23_lm)
    plt.draw()
    plt.pause(.001)
    input("Press [enter] to continue.")
    
    
    img34 = displayMatches(gr3,kp3,gr_curr,kp_curr,matches34,mask_RP_34, False)
    
    plt.imshow(img34)
    plt.title('Image 3 to 4')
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    
    print("Frame3_to_lm: ",len(frame4_to_lm))
    
    landmarks_34 = np.array([landmarks_23_new[frame4_to_lm[k]] for k in 
                            frame4_to_lm.keys()])
    
    lm_kps_4 = np.array([kp_curr[k].pt for k in frame4_to_lm.keys()])
    success, T_3_4, inliers = T_from_PNP(landmarks_34, lm_kps_4, K, D)
    
    plt.figure(2)
    plt.title('Image 3 to 4 PNP')
    graph = plot_3d_points(ax2, landmarks_34, linestyle="", marker="o", color='r')
    plot_pose3_on_axes(ax2, T_3_4, axis_length=2.0)
    
    if CHESSBOARD:
        ret4, corners4 = cv2.findChessboardCorners(gr_curr, (16,9),None)
        corners4_ud = cv2.undistortPoints(corners4,K,D)
    
        corners_34 = triangulate(T_2_3, T_3_4, corners3_ud, corners4_ud)
        graph = plot_3d_points(ax2, corners_34, linestyle="", marker=".",color='black')
    
    set_axes_equal(ax2)             # important!
    plt.draw()
    plt.pause(0.01)
    input("Press [enter] to continue.")
    landmarks_34_new = triangulate(T_2_3, T_3_4, kp3_match_34_ud[mask_RP_34[:,0]==1], 
                                                 kp4_match_34_ud[mask_RP_34[:,0]==1])
    graph = plot_3d_points(ax2, landmarks_34_new, linestyle="", marker="o", color='g')
    set_axes_equal(ax2)             # important!
    plt.title('Image 2 to 3 New Landmarks')
    plt.draw()
    plt.pause(0.01)
    input("Press [enter] to continue.")
    
    plt.close(fig='all')

process_frame(img4, mask4)