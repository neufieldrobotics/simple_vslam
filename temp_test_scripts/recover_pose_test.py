"""
recover_Pose_test

@author: vik748
"""
import sys,os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

# This allows adding correct path whether run from file, spyder or notebook
try:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
except NameError as e:
    print(e)
    this_file_dir = globals()['_dh'][0]
    
sys.path.insert(0, os.path.join(this_file_dir, os.pardir,'helper_functions'))
from vslam_helper import *

data_path = os.path.join(this_file_dir,os.pardir,'data')
img_folder = 'gopro_chess_board_800x600'

img_path1 = os.path.join(data_path, img_folder,'GOPR1550.jpg')
img_path2 = os.path.join(data_path, img_folder,'GOPR1552.jpg')

gr1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
gr2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

# Convert images to grayscale
#gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

chess_board_corners_file = os.path.join(data_path, img_folder, 'gopro_chess_board_800x600_corners.yaml')

with open(chess_board_corners_file) as file:
    corners_dict = yaml.safe_load(file)

'''
K = np.array([[700.551256,   0.     ,  403.995336],
              [  0.      , 695.41896,  289.95235 ],
              [  0.      ,   0.     ,    1.0     ]])
 
D = np.array([[-0.28571128, 0.16130412, 0.00005361, -0.00014855, -0.07717838]])
'''
K = np.array([[7.05603952310279169e+02, 0., 4.05317502366967972e+02],
              [0., 6.98401143710530164e+02, 2.97246118900126248e+02],
              [  0.      ,   0.     ,    1.0     ]])
 
D = np.array([[-2.77873196815023482e-01, 1.23933030802774416e-01, -4.01211313040891011e-04, -2.43040177514681940e-04]])


print(K,D)

# Make masks to mask the chess board from the image matching
mask_pts_1 = np.array(corners_dict[os.path.basename(img_path1)])
mask_pts_2 = np.array(corners_dict[os.path.basename(img_path2)])

mask = np.ones(gr1.shape[:2], dtype=np.uint8)
mask1 = cv2.fillConvexPoly(mask.copy(), mask_pts_1, color=[0, 0, 0])
mask2 = cv2.fillConvexPoly(mask.copy(), mask_pts_2, color=[0, 0, 0])


#Initiate ORB detector
detector = cv2.ORB_create(nfeatures=25000, edgeThreshold=31, patchSize=31, nlevels=8, 
                          fastThreshold=15, scaleFactor=1.2, WTA_K=2,
                          scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

# find the keypoints and descriptors with ORB
kp1, des1 = detector.detectAndCompute(gr1, mask1)
kp2, des2 = detector.detectAndCompute(gr2, mask2)

TILE_KP = True
RADIAL_NON_MAX = True
RADIAL_NON_MAX_RADIUS = 3
TILEY = 4
TILEX = 3

kp1 = detector.detect(gr1,mask1)
kp2 = detector.detect(gr2,mask2)
print ("Points detected: ",len(kp1), " and ", len(kp2))

if TILE_KP:
    kp1 = tiled_features(kp1, gr1.shape, TILEY, TILEX)
    kp2 = tiled_features(kp2, gr2.shape, TILEY, TILEX)
    print ("Points after tiling supression: ",len(kp1), " and ", len(kp2))

if RADIAL_NON_MAX:
    kp1 = radial_non_max_kd(kp1,RADIAL_NON_MAX_RADIUS)
    kp2 = radial_non_max_kd(kp2,RADIAL_NON_MAX_RADIUS)
    print ("Points after radial supression: ",len(kp1), " and ", len(kp2))

kp1, des1 = detector.compute(gr1,kp1)
kp2, des2 = detector.compute(gr2,kp2)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

#matches = bf.match(des1,des2)
matches = knn_match_and_lowe_ratio_filter(bf, des1, des2,threshold=0.9, dist_mask_12=None, draw_plot_dist=False)
kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])

kp1_match_ud = cv2.undistortPoints(np.expand_dims(kp1_match,axis=1),K,D)
kp2_match_ud = cv2.undistortPoints(np.expand_dims(kp2_match,axis=1),K,D)

E, mask_e = cv2.findEssentialMat(kp1_match_ud, kp2_match_ud, focal=1.0, pp=(0., 0.), 
                                 method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(matches),"matches")

points, rot_2R1, trans_2t1, mask_RP = cv2.recoverPose(E, kp1_match_ud, kp2_match_ud, mask=mask_e)
print("points:",points,"\trecover pose mask:",np.sum(mask_RP!=0))
print("R:",rot_2R1,"t:",trans_2t1.T)

bool_mask = mask_RP.astype(bool)
img_valid = cv2.drawMatches(gr1,kp1,gr2,kp2,matches, None, 
                            matchColor=(0, 255, 0), 
                            matchesMask=bool_mask.ravel().tolist(), flags=2)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
title = ax1.set_title('Feature Matching')
ax1.imshow(img_valid)
plt.pause(0.1)

# Find chess board corners to see how well the poses are estimated
ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)

corners1_ud = cv2.undistortPoints(corners1,K,D)
corners2_ud = cv2.undistortPoints(corners2,K,D)

'''
#Create 3 x 4 Homogenous Transform
Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
print ("Pose_1: ", Pose_1)
Pose_2 = np.hstack((R, t))
print ("Pose_2: ", Pose_2)

# Points Given in N,1,2 array 
landmarks_hom = cv2.triangulatePoints(Pose_1, Pose_2, 
                                      kp1_match_ud[mask_RP[:,0]==1], 
                                      kp2_match_ud[mask_RP[:,0]==1]).T
landmarks_hom_norm = landmarks_hom /  landmarks_hom[:,-1][:,None]
landmarks = landmarks_hom_norm[:, :3]

corners_hom = cv2.triangulatePoints(Pose_1, Pose_2, corners1_ud, corners2_ud).T
corners_hom_norm = corners_hom /  corners_hom[:,-1][:,None]
corners_12 = corners_hom_norm[:, :3]
'''
#t = -t
pose_wT1 = np.eye(4)
pose_2T1 = compose_T(rot_2R1,trans_2t1)
pose_1T2 = T_inv(pose_2T1)
pose_wT2 = pose_wT1 @ pose_1T2

print ("Pose_1: ", pose_wT1)
print ("Pose_2: ", pose_2T1)

landmarks, _ = triangulate(pose_wT1, pose_wT2, 
                           kp1_match_ud[mask_RP[:,0]==1], 
                           kp2_match_ud[mask_RP[:,0]==1] )

corners_12 ,_ = triangulate(pose_wT1, pose_wT2, corners1_ud, corners2_ud )

fig2, ax2 = initialize_3d_plot(number=2, title='Chess board reconstructed from calculated poses', view=(-70, -90))


# Plot triangulated featues in Red
graph = plot_3d_points(ax2, landmarks, linestyle="", marker="o",color='r')
#graph, = ax2.plot(landmarks[:,0], landmarks[:,1], landmarks[:,2], linestyle="", marker="o",color='r')
# Plot triangulated chess board in Green
graph = plot_3d_points(ax2, corners_12, linestyle="", marker="o",color='g')
#graph, = ax2.plot(corners_12[:,0], corners_12[:,1], corners_12[:,2], linestyle="", marker=".",color='g')

# Plot pose 1
plot_pose3_on_axes(ax2, pose_wT1, axis_length=0.5)
#Plot pose 2
plot_pose3_on_axes(ax2, pose_wT2, axis_length=1.0)
ax2.set_zlim3d(-2,5)
set_axes_equal(ax2)
ax2.view_init(-70, -90)  
plt.draw()
plt.pause(0.5)
input("Press [enter] to continue.")
