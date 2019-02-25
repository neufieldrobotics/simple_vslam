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
PAUSES = False
paused = False
cue_to_exit = False

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
    clahe = cv2.createCLAHE(**config_dict['CLAHE_settings'])
    gr1 = clahe.apply(gr1)
    gr2 = clahe.apply(gr2)

#def onClick(event):
#    print("Click")

def onKey(event):
    global paused, cue_to_exit
    #print('you pressed', event.key, event.xdata, event.ydata)
    if event.key==" ":
        paused = not paused
    if event.key=="q":
        cue_to_exit = True

            
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
'''
lk_params = dict( winSize  = (65,65),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
'''
kp1_match_12  = np.expand_dims(np.array([o.pt for o in kp1],dtype='float32'),1)
kp2_match_12, mask_klt, err = cv2.calcOpticalFlowPyrLK(gr1, gr2, kp1_match_12, None, **config_dict['KLT_settings'])
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
#fig2.canvas.mpl_connect('button_press_event', onClick)
fig2.canvas.mpl_connect('key_press_event', onKey)


plt.draw()
plt.pause(.001)
input("Press [enter] to continue.")

kp2 = detector.detect(gr2,mask2)
print ("KP2 Points detected: ",len(kp2))

if TILE_KP:
    kp2 = tiled_features(kp2, gr2.shape, TILEY, TILEX)
    print ("Points after tiling supression: ",len(kp2))

if RADIAL_NON_MAX:
    kp2 = radial_non_max(kp2,RADIAL_NON_MAX_RADIUS)
    print ("Points after radial supression: ",len(kp2))

kp2_pts  = np.expand_dims(np.array([o.pt for o in kp2],dtype='float32'),1)
     
kp2_pts = remove_redundant_newkps(kp2_pts, kp2_match_12, 5)

print ("Points after redudancy check with current kps: ",len(kp2_pts))

img12_newpts = draw_points(img12,kp2_pts[:,0,:], color=[255,255,0])
fig1_image.set_data(img12_newpts)
fig1.canvas.draw_idle() #plt.pause(0.001)
if PAUSES: input("Press [enter] to continue.")
                    

'''
PROCESS FRAME
'''
frame_no = 3
def process_frame(img_curr, mask_curr, gr_prev, kp_prev_matchpc, kp_prev_cand, lm_prev, T_prev, corners_prev_ud):
    global frame_no
    gr_curr = cv2.cvtColor(img_curr,cv2.COLOR_BGR2GRAY)
    if USE_CLAHE: gr_curr = clahe.apply(gr_curr)
    
    kp_curr_matchpc, mask_klt, _ = cv2.calcOpticalFlowPyrLK(gr_prev, gr_curr, 
                                                            kp_prev_matchpc, None, **config_dict['KLT_settings'])
    kp_curr_cand_matchpc, mask_cand_klt, _ = cv2.calcOpticalFlowPyrLK(gr_prev, gr_curr, 
                                                                      kp_prev_cand, None, **config_dict['KLT_settings'])

    print ("KLT Tracked: ",np.sum(mask_klt)," of total ",len(kp_prev_matchpc),"keypoints")
    print ("KLT Candidates Tracked: ",np.sum(mask_cand_klt)," of total ",len(kp_prev_cand),"keypoints")
 
    
    kp_prev_matchpc = kp_prev_matchpc[mask_klt[:,0]==1]
    kp_curr_matchpc = kp_curr_matchpc[mask_klt[:,0]==1]
    kp_prev_cand = kp_prev_cand[mask_cand_klt[:,0]==1]
    kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_cand_klt[:,0]==1]
    lm_prev = lm_prev[mask_klt[:,0]==1]
    
    kp_prev_all = np.vstack((kp_prev_matchpc, kp_prev_cand))
    kp_curr_all = np.vstack((kp_curr_matchpc, kp_curr_cand_matchpc))
    
    #kp_prev_matchpc_ud = cv2.undistortPoints(kp_prev_matchpc,K,D)
    #kp_curr_matchpc_ud = cv2.undistortPoints(kp_curr_matchpc,K,D)

    kp_prev_all_ud = cv2.undistortPoints(kp_prev_all,K,D)
    kp_curr_all_ud = cv2.undistortPoints(kp_curr_all,K,D)

    E, mask_e_all = cv2.findEssentialMat(kp_prev_all_ud, kp_curr_all_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, prob=0.999, threshold=0.001)
    essen_mat_pts = np.sum(mask_e_all)    
    
    print ("Essential matrix: used ",essen_mat_pts ," of total ",len(kp_curr_all),"matches")
    '''
    _, _, _, mask_RP_all = cv2.recoverPose(E, kp_prev_all_ud, kp_curr_all_ud, np.eye(3), 100.0, mask=mask_e_all)
    '''
    mask_RP_all = mask_e_all
    print ("Recover pose: used ",np.sum(mask_RP_all) ," of total ",essen_mat_pts," matches")
    
    mask_RP_feat = mask_RP_all[:len(kp_prev_matchpc)]
    mask_RP_cand = mask_RP_all[-len(kp_prev_cand):]
    
    if mask_curr is not None:
        gr_curr_masked = cv2.addWeighted(mask_curr, 0.2, gr_curr, 1 - 0.2, 0)
    else: gr_curr_masked = gr_curr
    
    img_track_feat = draw_point_tracks(gr_prev,kp_prev_matchpc,gr_curr_masked,kp_curr_matchpc,mask_RP_feat, False)
    img_track_all = draw_point_tracks(gr_prev,kp_prev_cand,img_track_feat,kp_curr_cand_matchpc,mask_RP_cand, False, color=[255,255,0])

    fig1_image.set_data(img_track_all)
    fig1.canvas.draw_idle(); plt.pause(0.01)
    if PAUSES: input("Press [enter] to continue.")
        
    success, T_cur, inliers = T_from_PNP(lm_prev[mask_RP_feat[:,0]==1], 
                                         kp_curr_matchpc[mask_RP_feat==1], K, D)
    if not success:
        print ("PNP faile in frame ",frame_no," Exiting...")
        exit()
                       
    if CHESSBOARD:
        ret, corners_curr = cv2.findChessboardCorners(gr_curr, (16,9),None)
        corners_curr_ud = cv2.undistortPoints(corners_curr,K,D)
    
        corners = triangulate(T_prev, T_cur, corners_prev_ud, corners_curr_ud)
        graph = plot_3d_points(ax2, corners, linestyle="", marker=".", markersize=2, 
                               color='black' if frame_no%2==0 else 'orange')
    else:
        corners_curr_ud = None
    lm_updated = lm_prev[mask_RP_feat[:,0]==1]
    kp_curr_matchpc = kp_curr_matchpc[mask_RP_feat[:,0]==1]    
    
    kp_prev_cand_ud = kp_prev_all_ud[-len(kp_prev_cand):]
    kp_curr_cand_ud = kp_curr_all_ud[-len(kp_prev_cand):]
    
    #print("kp_prev_cand_ud: ", kp_prev_cand_ud.shape, " kp_curr_cand_ud: ",kp_curr_cand_ud.shape)

    kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_RP_cand[:,0]==1]    
    kp_prev_cand_ud = kp_prev_cand_ud[mask_RP_cand[:,0]==1]
    kp_curr_cand_ud = kp_curr_cand_ud[mask_RP_cand[:,0]==1]
    
    lm_cand, mask_tri = triangulate(T_prev, T_cur, kp_prev_cand_ud, 
                                          kp_curr_cand_ud, None)
    #print(mask_tri.shape)
    print("Point rejected in triangulation: ", np.sum(mask_tri)," out of length: ", len(mask_tri))
    
    #lm_updated = lm_prev[mask_RP_feat[:,0]==1]
    kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_tri==1]

    graph_pnp = plot_3d_points(ax2, lm_cand, linestyle="", color='r', marker=".", markersize=2)
    plot_pose3_on_axes(ax2, T_cur, axis_length=2.0, center_plot=True)
    fig2.canvas.draw_idle(); plt.pause(0.01)
    if PAUSES: input("Press [enter] to continue.")
    
    graph_pnp.remove()
    graph_newlm = plot_3d_points(ax2, lm_cand, linestyle="", color='C0', marker=".", markersize=2)    
    fig2.canvas.draw_idle(); plt.pause(0.1)
    
    lm_updated = np.concatenate((lm_updated,lm_cand))
    kp_curr_matchpc = np.concatenate((kp_curr_matchpc,kp_curr_cand_matchpc))
    
    kp_curr_cand = detector.detect(gr_curr,mask_curr)
    print ("New feature candidates detected: ",len(kp_curr_cand))
    
    if TILE_KP:
        kp_curr_cand = tiled_features(kp_curr_cand, gr_curr.shape, TILEY, TILEX)
        print ("candidates points after tiling supression: ",len(kp_curr_cand))
    
    if RADIAL_NON_MAX:
        kp_curr_cand = radial_non_max(kp_curr_cand,RADIAL_NON_MAX_RADIUS)
        print ("candidates points after radial supression: ",len(kp_curr_cand))
    
    kp_curr_cand_pts  = np.expand_dims(np.array([o.pt for o in kp_curr_cand],dtype='float32'),1)
         
    kp_curr_cand_pts = remove_redundant_newkps(kp_curr_cand_pts, kp_curr_matchpc, RADIAL_NON_MAX_RADIUS)
    
    print ("Candidate points after redudancy check against current kps: ",len(kp_curr_cand_pts))
    
    img_cand_pts = draw_points(img_track_all,kp_curr_cand_pts[:,0,:], color=[255,255,0])
    fig1_image.set_data(img_cand_pts)
    fig1.canvas.draw_idle(); plt.pause(0.01)
    
    frame_no += 1
    return gr_curr, kp_curr_matchpc,  kp_curr_cand_pts, lm_updated, T_cur, corners_curr_ud

    print ("\n \n FRAME "+frame_no+" COMPLETE \n \n")

print ("\n \n FRAME 2 COMPLETE \n \n")

img3 = cv2.imread(images[init_imgs_indx[1]+img_step])

if USE_MASKS:
    mask = cv2.imread(masks[init_imgs_indx[1]+1],cv2.IMREAD_GRAYSCALE)
else:
    mask = None

out = process_frame(img3, mask, gr2, kp2_match_12, kp2_pts, landmarks_12, T_1_2, corners2_ud)
    
print ("\n \n FRAME 3 COMPLETE \n \n")

for i in range(init_imgs_indx[1]+img_step*2,len(images),img_step):
    while(paused):   
        print('.', end='', flush=True)
        #fig1.canvas.draw_idle()
        #fig2.canvas.draw_idle()
        plt.pause(0.1)
    if cue_to_exit: print("EXITING!!!"); break
    if USE_MASKS:
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
    print("Processing image: ",images[i])
    img = cv2.imread(images[i])
    st = time.time()
    out = process_frame(img, mask, *out)
    print("Time to process last frame: ",time.time()-st)
    plt.pause(0.001)
    print ("\n \n FRAME ", i ," COMPLETE \n \n")

plt.close(fig='all')