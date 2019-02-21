#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from vslam_helper import *
from ssc import *
import yaml
import glob
import re
import argparse
np.set_printoptions(precision=3,suppress=True)


parser = argparse.ArgumentParser(description='This is the simple VSLAM pipeline')
parser.add_argument('-c', '--config', help='location of config file in yaml format',
                    default='config/go_pro_icebergs_config.yaml') #go_pro_icebergs_config.yaml
args = parser.parse_args()
 
print (sys.platform)

# Inputs, images and camera info

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
    window_xadj = 0
    window_yadj = 45
else:
    path = '/home/vik748/'
    window_xadj = 65
    window_yadj = 430
    

'''    
img1 = cv2.imread(path+'data/chess_board3/GOPR1550.JPG',1)          # queryImage
img2 = cv2.imread(path+'data/chess_board3/GOPR1551.JPG',1)  
img3 = cv2.imread(path+'data/chess_board3/GOPR1552.JPG',1)  
img4 = cv2.imread(path+'data/chess_board3/GOPR1553.JPG',1)
img5 = cv2.imread(path+'data/chess_board3/GOPR1554.JPG',1)

mask_pts_1550 = np.array([[2119, 1318], [3409, 1332], [3400, 2184], [2107, 2218]])
mask_pts_1551 = np.array([[1863, 1326], [3181, 1338], [3176, 2213], [1855, 2224]])
mask_pts_1552 = np.array([[1488, 1281], [2818, 1301], [2801, 2170], [1474, 2178]])
mask_pts_1553 = np.array([[1247, 1286], [2562, 1295], [2556, 2178], [1224, 2164]])
mask_pts_1554 = np.array([[946, 1264], [2255, 1281], [2232, 2170], [917, 2139]])
mask = np.zeros(img1.shape[:2], dtype=np.uint8)

mask1 = 255 - cv2.fillConvexPoly(mask, mask_pts_1550, color=[255, 255, 255])
mask2 = 255 - cv2.fillConvexPoly(mask, mask_pts_1551, color=[255, 255, 255])
mask3 = 255 - cv2.fillConvexPoly(mask, mask_pts_1552, color=[255, 255, 255])
mask4 = 255 - cv2.fillConvexPoly(mask, mask_pts_1553, color=[255, 255, 255])
mask5 = 255 - cv2.fillConvexPoly(mask, mask_pts_1554, color=[255, 255, 255])
'''

# create a mask image filled with zeros, the size of original image
config_dict = yaml.load(open(args.config))
K = np.array(config_dict['K'])
D = np.array(config_dict['D'])
CHESSBOARD = config_dict['chessboard']
TILEY=config_dict['tiling_non_max_tile_y']; 
TILEX=config_dict['tiling_non_max_tile_x']; 
TILE_KP = config_dict['use_tiling_non_max_supression']
USE_MASKS = config_dict['use_masks']
RADIAL_NON_MAX = config_dict['radial_non_max']
RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
image_folder = config_dict['image_folder']
image_ext = config_dict['image_ext']
init_imgs_indx = config_dict['init_image_indxs']
img_step = config_dict['image_step']
PAUSES = False
USE_CLAHE = True

images = sorted([f for f in glob.glob(path+image_folder+'/*') 
                 if re.match('^.*\.'+image_ext+'$', f, flags=re.IGNORECASE)])

print(K,D)

img1 = cv2.imread(images[init_imgs_indx[0]])
img2 = cv2.imread(images[init_imgs_indx[1]])
#mask = 255 - np.zeros(img1.shape[:2], dtype=np.uint8)

if USE_MASKS:
    masks_folder = config_dict['masks_folder']
    masks_ext = config_dict['masks_ext']
    masks = sorted([f for f in glob.glob(path+masks_folder+'/*') 
                    if re.match('^.*\.'+masks_ext+'$', f, flags=re.IGNORECASE)])
    mask1 = cv2.imread(masks[init_imgs_indx[0]],cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(masks[init_imgs_indx[1]],cv2.IMREAD_GRAYSCALE)
else:
    mask1 = None
    mask2 = None

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,6))

if USE_CLAHE:
    gr1 = clahe.apply(gr1)
    gr2 = clahe.apply(gr2)

#Initiate ORB detector
detector = cv2.ORB_create(**config_dict['ORB_settings'])

#matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 1
matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 20,
                                   multi_probe_level = 2), dict(checks=100))

'''
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
if TILE_KP:
    kp1 = tiled_features(kp1, gr1.shape, TILEY, TILEX)
    kp2 = tiled_features(kp2, gr2.shape, TILEY, TILEX)
    print ("Points after tiling supression: ",len(kp1))

if RADIAL_NON_MAX:
    kp1 = radial_non_max(kp1,RADIAL_NON_MAX_RADIUS)
    kp2 = radial_non_max(kp2,RADIAL_NON_MAX_RADIUS)
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
fig1_image = plt.imshow(img12)
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
fig3_image = plt.imshow(img2_track)
plt.title('Image 1 to 2 matches')
plt.axis("off")
fig3.subplots_adjust(0,0,1,1)
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

landmarks_12, mask_RP_12 = triangulate(np.eye(4), T_1_2, kp1_match_12_ud, 
                                       kp2_match_12_ud, mask_RP_12[:,0])

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
fig2.subplots_adjust(0,0,1,1)
plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
#move_figure(position="right")
ax2.set_aspect('equal')         # important!
fig2.suptitle('Image 1 to 2 after triangulation')

graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker=".", markersize=2)

if CHESSBOARD:
    ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
    ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)
    
    corners1_ud = cv2.undistortPoints(corners1,K,D)
    corners2_ud = cv2.undistortPoints(corners2,K,D)
   
    corners_12 = triangulate(np.eye(4), T_1_2, corners1_ud, corners2_ud)
    graph = plot_3d_points(ax2, corners_12, linestyle="", color='g', marker=".", markersize=2)
else:
    corners2_ud = None
    
plot_pose3_on_axes(ax2, T_1_2, axis_length=1.0)
plot_pose3_on_axes(ax2,np.eye(4), axis_length=0.5)

set_axes_equal(ax2)
ax2.view_init(0, -90)

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

fig4 = plt.figure(4)
plt.draw()

'''
PROCESS FRAME
'''
frame_no = 3
def process_frame(img_curr, mask_curr, gr_prev, kp_prev, des_prev,frame_p2lm, 
                  matchespc_prev, lm_prev, T_prev, corners_prev_ud):
    #plt.imshow(mask_curr)
    plt.draw()
    global frame_no
    gr_curr = cv2.cvtColor(img_curr,cv2.COLOR_BGR2GRAY)
    if USE_CLAHE: gr_curr = clahe.apply(gr_curr)
    kp_curr = detector.detect(gr_curr,mask_curr)
    print ("Points detected: ",len(kp_curr))
    
    if TILE_KP:
        kp_curr = tiled_features(kp_curr, gr_curr.shape, TILEY, TILEX)
        print ("Points after tiling supression: ",len(kp1))

    if RADIAL_NON_MAX:
        kp_curr = radial_non_max(kp_curr,RADIAL_NON_MAX_RADIUS)
        print ("Points after radial supression: ",len(kp_curr))
    
    kp_curr, des_curr = detector.compute(gr_curr,kp_curr)
        
    #print(des_prev, des_curr)
    #des_curr = np.ascontiguousarray(des_curr)
    
    matchespc = matcher.match(des_prev,des_curr)
    kp_prev_matchpc = np.array([kp_prev[mat.queryIdx].pt for mat in matchespc])
    kp_curr_matchpc = np.array([kp_curr[mat.trainIdx].pt for mat in matchespc])
    
    kp_prev_matchpc_ud = cv2.undistortPoints(np.expand_dims(kp_prev_matchpc,axis=1),K,D)
    kp_curr_matchpc_ud = cv2.undistortPoints(np.expand_dims(kp_curr_matchpc,axis=1),K,D)
    
    E, mask_e = cv2.findEssentialMat(kp_prev_matchpc_ud, kp_curr_matchpc_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, prob=0.999, threshold=0.001)
    
    mask_e_copy = mask_e.copy()
    
    print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(matchespc),"matches")
    _, _, _, mask_RP = cv2.recoverPose(E, kp_prev_matchpc_ud, kp_curr_matchpc_ud,mask=mask_e)

    print ("Recover pose: used ",np.sum(mask_RP) ," of total ",len(matchespc),"matches")
    
    matchespc_filt = [matchespc[i] for i in range(len(matchespc)) if mask_RP[i]==1]
    frame_c2p = {mat.trainIdx:mat.queryIdx for mat in matchespc_filt}
    
    frame_c2lm = {id:frame_p2lm.get(frame_c2p[id]) 
                    for id in frame_c2p.keys() 
                    if frame_p2lm.get(frame_c2p[id]) is not None}
    '''
    To plot matches in 2-3 which are found in 4 etc.
    mask_lm_cinp = np.zeros(mask_RP_prev.shape)
    lm_2pf = dict([[v,k] for k,v in frame3_to_lm.items()])
    frame_p2_matchespc_prev = {mat.trainIdx:match_id for match_id,mat in enumerate(matchespc_prev)}


    for frame_ckp, lm_id in frame_c2lm.items():
        frame_pkp = lm_2pf[lm_id]
        matches_prev_id = frame_p2_matchespc_prev[frame_pkp]
        mask_lm_cinp[matches_prev_id]=1.0
    #plt.title('Image 1 to 2 - Landmarks found in 3')
    #img23_lm = displayMatches(gr2,kp2,gr3,kp_prev,matches23,mask_lm_cinp, False, in_image=img23, color=(255,165,0))
    #plt.imshow(img23_lm); plt.draw(); plt.pause(.001)
    #
    '''
    #fig1 = plt.figure(1)
    
    img_matches = displayMatches(gr_prev,kp_prev,gr_curr,kp_curr,matchespc,mask_RP, False)
    
    #plt.imshow(img_matches)
    fig1_image.set_data(img_matches)
    #plt.title('Current frame matches to prev'); 
    fig1.canvas.draw_idle() #plt.pause(0.001)
    #fig3 = plt.figure(3)
    img_track = draw_feature_tracks(gr_prev,kp_prev,gr_curr,kp_curr,matchespc,mask_RP)
    #plt.imshow(img_track)
    #plt.title('Image 1 to 2 matches'); plt.draw(); plt.pause(0.1)
    fig3_image.set_data(img_track)
    fig3.canvas.draw_idle()
    
    if PAUSES: input("Press [enter] to continue.")
    
    print("frame_c2lm: ",len(frame_c2lm))
    
    lm_cur = np.array([lm_prev[frame_c2lm[k]] for k in frame_c2lm.keys()])
    
    lm_cur_kps_in_frame = np.array([kp_curr[k].pt for k in frame_c2lm.keys()])
    success, T_cur, inliers = T_from_PNP(lm_cur, lm_cur_kps_in_frame, K, D)
    if not success:
        print ("PNP faile in frame ",frame_no," Exiting...")
        exit()
        
    print("PNP status: ", success)
    #plt.figure(2)
    #plt.title('Image prev to curr PNP pos and landmarks used')
    
    st = time.time()
    graph_pnp = plot_3d_points(ax2, lm_cur, linestyle="", color='r', marker=".", markersize=2)
    
    plot_pose3_on_axes(ax2, T_cur, axis_length=2.0, center_plot=True)
    fig2.canvas.draw_idle()
    if PAUSES: input("Press [enter] to continue.")
    
    if CHESSBOARD:
        ret, corners_curr = cv2.findChessboardCorners(gr_curr, (16,9),None)
        corners_curr_ud = cv2.undistortPoints(corners_curr,K,D)
    
        corners = triangulate(T_prev, T_cur, corners_prev_ud, corners_curr_ud)
        graph = plot_3d_points(ax2, corners, linestyle="", marker=".", markersize=2, 
                               color='black' if frame_no%2==0 else 'orange')
    else:
        corners_curr_ud = None
    
    mask_newpts = np.array([1 if (mask_RP[i,0]==1 and frame_c2lm.get(matchespc[i].trainIdx) is None) 
                           else 0 for i in range(len(kp_curr_matchpc_ud))],dtype='int')
    
    lm_cur_new, mask_newpts = triangulate(T_prev, T_cur, kp_prev_matchpc_ud, 
                                          kp_curr_matchpc_ud, mask_newpts)
    
    graph_newlm = plot_3d_points(ax2, lm_cur_new, linestyle="", color='g', marker=".", markersize=2)
    plt.title('Current frame New Landmarks'); set_axes_equal(ax2); plt.draw(); plt.pause(0.001)
    
    # color landmarks back to blue
    #graph = plot_3d_points(ax2, lm_cur, linestyle="", color='C0', marker=".", markersize=2)
    graph_pnp.remove()
    graph_newlm.remove()
    graph_newlm = plot_3d_points(ax2, lm_cur_new, linestyle="", color='C0', marker=".", markersize=2)    
    plt.title('Current frame landmaks added in'); set_axes_equal(ax2); plt.draw(); plt.pause(0.001)
    
    lm_newids = -np.ones(mask_newpts.shape[0],dtype=int)
    lm_newids[mask_newpts.ravel()==1]=np.arange(np.sum(mask_newpts))+len(lm_prev)
    
    # Create a dictionary {KP2 index of match : landmark number}
    frame_c2lm_new = {mat.trainIdx:lm_id for lm_id,mat in zip(lm_newids, matchespc)
                    if lm_id!=-1 }
    frame_c2lm.update(frame_c2lm_new)
    lm_updated = np.concatenate((lm_prev,lm_cur_new))
    print ("Adding ",len(frame_c2lm_new), " landmars. Total landmarks: ", lm_updated.shape[0])

    frame_no += 1
    return gr_curr, kp_curr, des_curr, frame_c2lm, matchespc, lm_updated, T_cur, corners_curr_ud

print ("\n \n FRAME 2 COMPLETE \n \n")

img3 = cv2.imread(images[init_imgs_indx[1]+img_step])

if USE_MASKS:
    mask = cv2.imread(masks[init_imgs_indx[1]+1],cv2.IMREAD_GRAYSCALE)
else:
    mask = None

out = process_frame(img3, mask, gr2, kp2, des2, frame2_to_lm, matches12, 
                     landmarks_12, T_1_2, corners2_ud)

print ("\n \n FRAME 3 COMPLETE \n \n")

for i in range(init_imgs_indx[1]+2,len(images),img_step):
    if USE_MASKS:
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(images[i])
    st = time.time()
    out = process_frame(img, mask, *out)
    print("Time to process last frame: ",time.time()-st)
    print ("\n \n FRAME ",i," COMPLETE \n \n")
    # press 'q' to exit

plt.close(fig='all')