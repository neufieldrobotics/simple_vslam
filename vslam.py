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
                    default='config/kitti_config.yaml') #go_pro_icebergs_config.yaml
args = parser.parse_args()
 
# Inputs, images and camera info

if sys.platform == 'darwin':
    path = '/Users/vik748/Google Drive/'
    window_xadj = 0
    window_yadj = 45
else:
    path = '/home/vik748/'
    window_xadj = 65
    window_yadj = 430
    
config_dict = yaml.load(open(args.config))
K = np.array(config_dict['K'])
D = np.array(config_dict['D'])
CHESSBOARD = config_dict['chessboard']
TILEY=config_dict['tiling_non_max_tile_y']; 
TILEX=config_dict['tiling_non_max_tile_x']; 
TILE_KP = config_dict['use_tiling_non_max_supression']
USE_MASKS = config_dict['use_masks']
USE_CLAHE = config_dict['use_clahe']
RADIAL_NON_MAX = config_dict['radial_non_max']
RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
image_folder = config_dict['image_folder']
image_ext = config_dict['image_ext']
init_imgs_indx = config_dict['init_image_indxs']
img_step = config_dict['image_step']
PAUSES = True

images = sorted([f for f in glob.glob(path+image_folder+'/*') 
                 if re.match('^.*\.'+image_ext+'$', f, flags=re.IGNORECASE)])
assert images is not None, "ERROR: No images read"

print(K,D)

img1 = cv2.imread(images[init_imgs_indx[0]])
img2 = cv2.imread(images[init_imgs_indx[1]])
#mask = 255 - np.zeros(img1.shape[:2], dtype=np.uint8)

if USE_MASKS:
    masks_folder = config_dict['masks_folder']
    masks_ext = config_dict['masks_ext']
    masks = sorted([f for f in glob.glob(path+masks_folder+'/*') 
                    if re.match('^.*\.'+masks_ext+'$', f, flags=re.IGNORECASE)])
    assert len(masks)==len(images), "ERROR: Number of masks not equal to number of images"
    mask1 = cv2.imread(masks[init_imgs_indx[0]],cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(masks[init_imgs_indx[1]],cv2.IMREAD_GRAYSCALE)
else:
    mask1 = None
    mask2 = None

gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

if USE_CLAHE:
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,6))
    gr1 = clahe.apply(gr1)
    gr2 = clahe.apply(gr2)

#Initiate ORB detector
detector = cv2.ORB_create(**config_dict['ORB_settings'])

# find the keypoints and descriptors with ORB
kp1 = detector.detect(gr1,mask1)
#kp2 = detector.detect(gr2,mask2)

print ("Points detected: ",len(kp1))

if TILE_KP:
    kp1 = tiled_features(kp1, gr1.shape, TILEY, TILEX)
    print ("Points after tiling supression: ",len(kp1))

if RADIAL_NON_MAX:
    kp1 = radial_non_max(kp1,RADIAL_NON_MAX_RADIUS)
    print ("Points after radial supression: ",len(kp1))

lk_params = dict( winSize  = (65,65),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

kp1_match_12  = np.expand_dims(np.array([o.pt for o in kp1],dtype='float32'),1)
kp2_match_12, mask_klt, err = cv2.calcOpticalFlowPyrLK(gr1, gr2, kp1_match_12, None, **lk_params)
print ("KLT tracked: ",np.sum(mask_klt) ," of total ",len(kp1_match_12),"keypoints")

kp1_match_12 = kp1_match_12[mask_klt[:,0]==1]
kp2_match_12 = kp2_match_12[mask_klt[:,0]==1]

kp1_match_12_ud = cv2.undistortPoints(kp1_match_12,K,D)
kp2_match_12_ud = cv2.undistortPoints(kp2_match_12,K,D)

E_12, mask_e_12 = cv2.findEssentialMat(kp1_match_12_ud, kp2_match_12_ud, focal=1.0, pp=(0., 0.), 
                                       method=cv2.RANSAC, prob=0.999, threshold=0.001)
print ("Essential matrix: used ",np.sum(mask_e_12) ," of total ",len(kp1_match_12),"matches")
essen_mat_pts = np.sum(mask_e_12)
points, R_21, t_21, mask_RP_12 = cv2.recoverPose(E_12, kp1_match_12_ud, kp2_match_12_ud,mask=mask_e_12)
print("points:",points,"\trecover pose mask:",np.sum(mask_RP_12!=0))
print ("Recover pose used ",np.sum(mask_RP_12) ," of total ",essen_mat_pts," matches in Essential matrix")
T_2_1 = compose_T(R_21,t_21)
T_1_2 = T_inv(T_2_1)
print("R:",R_21)
print("t:",t_21.T)

img12 = draw_point_tracks(gr1,kp1_match_12,gr2,kp2_match_12,mask_RP_12, False)

fig1 = plt.figure(1)
plt.get_current_fig_manager().window.setGeometry(window_xadj,window_yadj,640,338)

fig1_image = plt.imshow(img12)
plt.title('Image 1 to 2 matches')
plt.axis("off")
fig1.subplots_adjust(0,0,1,1)
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

# Trim the tracked key pts
kp1_match_12 = kp1_match_12[mask_RP_12[:,0]==1]
kp2_match_12 = kp2_match_12[mask_RP_12[:,0]==1]

kp1_match_12_ud = kp1_match_12_ud[mask_RP_12[:,0]==1]
kp2_match_12_ud = kp2_match_12_ud[mask_RP_12[:,0]==1]

landmarks_12, mask_tri_12 = triangulate(np.eye(4), T_1_2, kp1_match_12_ud, 
                                       kp2_match_12_ud, None)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
fig2.subplots_adjust(0,0,1,1)
plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
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

'''
lm_12 = -np.ones(mask_RP_12.shape[0],dtype=int)
lm_12[mask_RP_12.ravel()==1]=np.arange(np.sum(mask_RP_12))

# Create a dictionary {KP2 index of match : landmark number}
frame2_to_lm = {mat.trainIdx:lm_id for lm_id,mat in zip(lm_12, matches12)
                if lm_id!=-1 }
lm_to_frame2 = dict([[v,k] for k,v in frame2_to_lm.items()])
frame2_to_matches12 = {mat.trainIdx:match_id for match_id,mat in enumerate(matches12)}
'''

'''
PROCESS FRAME
'''
frame_no = 3
def process_frame(img_curr, mask_curr, gr_prev, kp_prev_matchpc, lm_prev, T_prev, corners_prev_ud):
    #plt.imshow(mask_curr)
    plt.draw()
    global frame_no
    gr_curr = cv2.cvtColor(img_curr,cv2.COLOR_BGR2GRAY)
    if USE_CLAHE: gr_curr = clahe.apply(gr_curr)
    
    kp_curr_matchpc, mask_klt, err = cv2.calcOpticalFlowPyrLK(gr_prev, gr_curr, kp_prev_matchpc, None, **lk_params)

    print ("KLT Tracked: ",np.sum(mask_klt)," of total ",len(kp_prev_matchpc),"keypoints")
    
    kp_prev_matchpc = kp_prev_matchpc[mask_klt[:,0]==1]
    kp_curr_matchpc = kp_curr_matchpc[mask_klt[:,0]==1]
    lm_prev = lm_prev[mask_klt[:,0]==1]
    
    kp_prev_matchpc_ud = cv2.undistortPoints(kp_prev_matchpc,K,D)
    kp_curr_matchpc_ud = cv2.undistortPoints(kp_curr_matchpc,K,D)

    E, mask_e = cv2.findEssentialMat(kp_prev_matchpc_ud, kp_curr_matchpc_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, prob=0.999, threshold=0.001)
    essen_mat_pts = np.sum(mask_e)    
    
    print ("Essential matrix: used ",essen_mat_pts ," of total ",len(kp_curr_matchpc),"matches")
    _, _, _, mask_RP = cv2.recoverPose(E, kp_prev_matchpc_ud, kp_curr_matchpc_ud,mask=mask_e)

    print ("Recover pose: used ",np.sum(mask_RP) ," of total ",essen_mat_pts," matches")

    img_track = draw_point_tracks(gr_prev,kp_prev_matchpc,gr_curr,kp_curr_matchpc,mask_RP, False)
    fig1_image.set_data(img_track)
    fig1.canvas.draw_idle() #plt.pause(0.001)
    if PAUSES: input("Press [enter] to continue.")
    
    print("mask_RP: ",mask_RP.shape," KP_curr", kp_curr_matchpc[mask_RP==1].shape, " lm_prev: ", lm_prev[mask_RP[:,0]==1].shape)
    
    success, T_cur, inliers = T_from_PNP(lm_prev[mask_RP[:,0]==1], 
                                         kp_curr_matchpc[mask_RP==1], K, D)
    if not success:
        print ("PNP faile in frame ",frame_no," Exiting...")
        exit()
        
    print("PNP status: ", success)
       
    st = time.time()
    #graph_pnp = plot_3d_points(ax2, lm_cur, linestyle="", color='r', marker=".", markersize=2)
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
    lm_updated = lm_prev[mask_RP[:,0]==1]
    kp_curr_matchpc = kp_curr_matchpc[mask_RP[:,0]==1]    
        
    '''
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
    '''

    frame_no += 1
    return gr_curr, kp_curr_matchpc,  lm_updated, T_cur, corners_curr_ud

    print ("\n \n FRAME "+frame_no+" COMPLETE \n \n")

print ("\n \n FRAME 2 COMPLETE \n \n")

img3 = cv2.imread(images[init_imgs_indx[1]+img_step])

if USE_MASKS:
    mask = cv2.imread(masks[init_imgs_indx[1]+1],cv2.IMREAD_GRAYSCALE)
else:
    mask = None

out = process_frame(img3, mask, gr2, kp2_match_12, landmarks_12, T_1_2, corners2_ud)
    
print ("\n \n FRAME 3 COMPLETE \n \n")

for i in range(init_imgs_indx[1]+img_step+1,len(images),img_step):
    if USE_MASKS:
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(images[i])
    st = time.time()
    out = process_frame(img, mask, *out)
    print("Time to process last frame: ",time.time()-st)
    # press 'q' to exit

plt.close(fig='all')