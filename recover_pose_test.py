"""
recover_Pose_test

@author: vik748
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pose3_on_axes(axes, gRp, origin, axis_length=0.1):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # get rotation and translation (center)
    #gRp = pose.rotation().matrix()  # rotation from pose to global
    #t = pose.translation()
    #origin = np.array([t.x(), t.y(), t.z()])

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin, x_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin, y_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin, z_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

img1 = cv2.imread('/home/vik748/data/chess_board/GOPR1488.JPG',1)          # queryImage
img2 = cv2.imread('/home/vik748/data/chess_board/GOPR1490.JPG',1)  

fx = 3551.342810
fy = 3522.689669
cx = 2033.513326
cy = 1455.489194

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])

D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);

print(K,D)

# Convert images to greyscale
gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#Initiate ORB detector
detector = cv2.ORB_create(nfeatures=25000, edgeThreshold=15, patchSize=125, nlevels=32, 
                     fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)

# find the keypoints and descriptors with ORB
kp1, des1 = detector.detectAndCompute(gr1,None)
kp2, des2 = detector.detectAndCompute(gr2,None)

print ("Points detected: ",len(kp1), " and ", len(kp2))

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])

kp1_match_ud = cv2.undistortPoints(np.expand_dims(kp1_match,axis=1),K,D)
kp2_match_ud = cv2.undistortPoints(np.expand_dims(kp2_match,axis=1),K,D)

E, mask_e = cv2.findEssentialMat(kp1_match_ud, kp2_match_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(matches),"matches")

points, R, t, mask_RP = cv2.recoverPose(E, kp1_match_ud, kp2_match_ud, mask=mask_e)
print("points:",points,"\trecover pose mask:",np.sum(mask_RP!=0))
print("R:",R,"t:",t.T)

bool_mask = mask_RP.astype(bool)
img_valid = cv2.drawMatches(gr1,kp1,gr2,kp2,matches, None, 
                            matchColor=(0, 255, 0), 
                            matchesMask=bool_mask.ravel().tolist(), flags=2)

plt.imshow(img_valid)
plt.show()

ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)

corners1_ud = cv2.undistortPoints(corners1,K,D)
corners2_ud = cv2.undistortPoints(corners2,K,D)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
ax.set_zlim3d(-5,10)

# Plot triangulated featues in Red
graph, = ax.plot(landmarks[:,0], landmarks[:,1], landmarks[:,2], linestyle="", marker="o",color='r')
# Plot triangulated chess board in Green
graph, = ax.plot(corners_12[:,0], corners_12[:,1], corners_12[:,2], linestyle="", marker=".",color='g')

# Plot pose 1
plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=0.5)
#Plot pose 2
plot_pose3_on_axes(ax, R, t.T, axis_length=1.0)
ax.set_zlim3d(-2,5)
#set_axes_equal(ax)
ax.view_init(-70, -90)
plt.show()