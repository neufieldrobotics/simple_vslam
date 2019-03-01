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
import traceback
np.set_printoptions(precision=3,suppress=True)
import multiprocessing as mp
from colorama import Fore, Style
from itertools import cycle

'''
PROCESS FRAME
'''
frame_no = 3
def process_frame(gr_curr, mask_curr, kp_curr_cand_pts, gr_prev, kp_prev_matchpc, kp_prev_cand, lm_prev, T_prev):
    print ("len of lm_prev:",len(lm_prev))
    print ("len of kp_prev_matchpc:",len(kp_prev_matchpc))
    global frame_no, cam_pose_trail, cam_trail_pts, cam_pose, new_lm_graph
    time_start = time.time()

    kp_curr_matchpc, mask_klt, _ = cv2.calcOpticalFlowPyrLK(gr_prev, gr_curr, 
                                    kp_prev_matchpc, None, **config_dict['KLT_settings'])

    print ("KLT Tracked: ",np.sum(mask_klt)," of total ",len(kp_prev_matchpc),"keypoints")
    kp_prev_matchpc = kp_prev_matchpc[mask_klt[:,0].astype(bool)]
    kp_curr_matchpc = kp_curr_matchpc[mask_klt[:,0].astype(bool)]

    
    if len(kp_prev_cand)>0:
        kp_curr_cand_matchpc, mask_cand_klt, _ = cv2.calcOpticalFlowPyrLK(gr_prev, gr_curr, 
                                                            kp_prev_cand, None, **config_dict['KLT_settings'])
        print ("KLT Candidates Tracked: ",np.sum(mask_cand_klt)," of total ",len(kp_prev_cand),"keypoints")
        kp_prev_cand = kp_prev_cand[mask_cand_klt[:,0].astype(bool)]
        kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_cand_klt[:,0].astype(bool)]
    else: 
        kp_curr_cand_matchpc = np.zeros([0,1,2])
        print ("No New KLT Candidates Tracked. ")
        
    print("Time elapsed in optical flow: ",time.time()-time_start); time_start = time.time()

    # Filter out points that are not tracked
    #print("kp_prev_cand: ",kp_prev_cand)
    #print("kp_curr_cand_matchpc: ",kp_curr_cand_matchpc)

    lm_prev = lm_prev[mask_klt[:,0].astype(bool)]
    
    # Merge tracked LM and candidate pts to filter together in findEssential
    kp_prev_all = np.vstack((kp_prev_matchpc, kp_prev_cand))
    kp_curr_all = np.vstack((kp_curr_matchpc, kp_curr_cand_matchpc))
    
    kp_prev_all_ud = cv2.undistortPoints(kp_prev_all,K,D)
    kp_curr_all_ud = cv2.undistortPoints(kp_curr_all,K,D)

    print("Time elapsed in undistort: ",time.time()-time_start); time_start = time.time()

    E, mask_e_all = cv2.findEssentialMat(kp_prev_all_ud, kp_curr_all_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, prob=0.999, threshold=0.001)
    essen_mat_pts = np.sum(mask_e_all)  
    print("Time elapsed in find E: ",time.time()-time_start); time_start = time.time()

    
    print ("Essential matrix: used ",essen_mat_pts ," of total ",len(kp_curr_all),"matches")
    
    if FILTER_RP:
        # Recover Pose filtering is breaking under certain conditions. Leave out for now.
        _, _, _, mask_RP_all = cv2.recoverPose(E, kp_prev_all_ud, kp_curr_all_ud, np.eye(3), 100.0, mask=mask_e_all)
        print ("Recover pose: used ",np.sum(mask_RP_all) ," of total ",essen_mat_pts," matches")
    else: 
        mask_RP_all = mask_e_all
    
    # Split the combined mask to lm features and candidates
    mask_RP_feat = mask_RP_all[:len(kp_prev_matchpc)]
    mask_RP_cand = mask_RP_all[-len(kp_prev_cand):]
    
    # Display translucent mask on image.
    if mask_curr is not None:
        gr_curr_masked = cv2.addWeighted(mask_curr, 0.2, gr_curr, 1 - 0.2, 0)
    else: gr_curr_masked = gr_curr
    
    img_track_feat = draw_point_tracks(kp_prev_matchpc,gr_curr_masked,
                                       kp_curr_matchpc,mask_RP_feat, False)
    if len(kp_curr_cand_matchpc)>0:
        img_track_all = draw_point_tracks(kp_prev_cand,img_track_feat,
                                          kp_curr_cand_matchpc,mask_RP_cand, False, 
                                          color=[255,255,0])
    else: img_track_all = img_track_feat
    print("Time elapsed in drawing tracks: ",time.time()-time_start);time_start = time.time()


    #fig1_image.set_data(img_track_all)
    #fig1.canvas.draw_idle(); plt.pause(0.01)
    #if PAUSES: input("Press [enter] to continue.")

    lm_updated = lm_prev[mask_RP_feat[:,0].astype(bool)]
    print("LM prev was: ",len(lm_prev), " LM updated after Ess mat filter is :",len(lm_updated))
    kp_curr_matchpc = kp_curr_matchpc[mask_RP_feat[:,0].astype(bool)]
           
    success, T_cur, inliers = T_from_PNP(lm_updated, kp_curr_matchpc, K, D)
    print("Time elapsed in PNP: ",time.time()-time_start)

    print("PNP inliers: ",len(inliers), " of ",len(lm_updated))
    if not success:
        print ("PNP faile in frame ",frame_no," Exiting...")
        exit()
                       
    
    kp_prev_cand_ud = kp_prev_all_ud[-len(kp_prev_cand):]
    kp_curr_cand_ud = kp_curr_all_ud[-len(kp_prev_cand):]
    
    if len(kp_prev_cand)>0:
        kp_prev_cand = kp_prev_cand[mask_RP_cand[:,0].astype(bool)]    
        kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_RP_cand[:,0].astype(bool)]    
    kp_prev_cand_ud = kp_prev_cand_ud[mask_RP_cand[:,0].astype(bool)]
    kp_curr_cand_ud = kp_curr_cand_ud[mask_RP_cand[:,0].astype(bool)]
    
    lm_cand, mask_tri = triangulate(T_prev, T_cur, kp_prev_cand_ud, 
                                    kp_curr_cand_ud, None)
    print("Time elapsed in triangulate: ",time.time()-time_start); time_start = time.time()

    print("Point after rejection in triangulation: ", np.sum(mask_tri)," out of length: ", len(mask_tri))
    
    if len(kp_prev_cand)>0:
        img_rej_pts = draw_point_tracks(kp_prev_cand, img_track_all, 
                                        kp_curr_cand_matchpc, 1-mask_tri, False, color=[255,0,0])
    else: img_rej_pts = img_track_all
    print("Time elapsed in draw pt tracks: ",time.time()-time_start); time_start = time.time()

    #fig1_image.set_data(img_rej_pts)
    #fig1.canvas.draw_idle(); #plt.pause(0.01)
    #print("Time elapsed in draw pt track SET DATA: ",time.time()-time_start)

    #lm_updated = lm_prev[mask_RP_feat[:,0]==1]
    kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_tri[:,0].astype(bool)]
    
    try:    
        new_lm_graph.remove()
    except NameError:
        pass
    new_lm_graph = plot_3d_points(ax2, lm_cand, linestyle="", color='r', marker=".", markersize=2)
    plot_pose3_on_axes(ax2, T_cur, axis_length=2.0, center_plot=True, line_obj_list=cam_pose)
    
    cam_trail_pts = np.append(cam_trail_pts,T_cur[:3,[-1]].T,axis=0)
    cam_pose_trail = plot_3d_points(ax2,cam_trail_pts , line_obj=cam_pose_trail, linestyle="", color='g', marker=".", markersize=2)
    fig2.canvas.draw_idle(); #plt.pause(0.01)
    if PAUSES: input("Press [enter] to continue.")

    # Remove pose lines and landmarks to speed plotting
    if PLOT_LANDMARKS:
        graph_newlm = plot_3d_points(ax2, lm_cand, linestyle="", color='C0', marker=".", markersize=2)    
        fig2.canvas.draw_idle(); #plt.pause(0.01)
    
    print("len of lm updated ", len(lm_updated))
    lm_updated = np.concatenate((lm_updated,lm_cand))
    print("len of lm updated again ", len(lm_updated))
    kp_curr_matchpc = np.concatenate((kp_curr_matchpc,kp_curr_cand_matchpc))
    
    print("Time elapsed in beofre orb: ",time.time()-time_start)
    '''
    kp_curr_cand = detector.detect(gr_curr,mask_curr)
    print ("New feature candidates detected: ",len(kp_curr_cand))
    
    if TILE_KP:
        kp_curr_cand = tiled_features(kp_curr_cand, gr_curr.shape, TILEY, TILEX)
        print ("candidates points after tiling supression: ",len(kp_curr_cand))
    
    if RADIAL_NON_MAX:
        kp_curr_cand = radial_non_max(kp_curr_cand,RADIAL_NON_MAX_RADIUS)
        print ("candidates points after radial supression: ",len(kp_curr_cand))
    
    kp_curr_cand_pts  = np.expand_dims(np.array([o.pt for o in kp_curr_cand],dtype='float32'),1)
    '''     
    kp_curr_cand_pts = remove_redundant_newkps(kp_curr_cand_pts, kp_curr_matchpc, RADIAL_NON_MAX_RADIUS)
    
    print("Time elapsed in orb, filtering, radial: ",time.time()-time_start)

    print ("Candidate points after redudancy check against current kps: ",len(kp_curr_cand_pts))
    img_cand_pts = draw_points(img_rej_pts,kp_curr_cand_pts[:,0,:], color=[255,255,0])
    fig1_image.set_data(img_cand_pts)
    fig1.canvas.draw_idle(); plt.pause(0.05)
    
    frame_no += 1
    
    print ("FRAME deq "+str(frame_no)+" COMPLETE")

    return gr_curr, kp_curr_matchpc,  kp_curr_cand_pts, lm_updated, T_cur

def preprocess_frame(image_name, detector, mask_name=None, clahe_obj=None, tiling=None, rnm_radius=None):
    print("Pre-processing image: "+image_name)

    pt = time.time()
    img = cv2.imread(image_name)
    gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if mask_name is not None:
        mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
    else: mask= None

    if clahe_obj is not None: gr = clahe_obj.apply(gr)
    
    kp = detector.detect(gr,mask)
    pbf = "New feature candidates detected: "+str(len(kp))
    
    if tiling is not None:
        kp = tiled_features(kp, gr.shape, *tiling)
        pbf += " > tiling supression: "+str(len(kp))
    
    if rnm_radius is not None:
        kp = radial_non_max(kp,rnm_radius)
        pbf += " > radial supression: "+str(len(kp))
    
    kp_pts = np.expand_dims(np.array([o.pt for o in kp],dtype='float32'),1)
    print(pbf+"\nPre-processing time is", time.time()-pt)
    return gr, mask, kp_pts

def writer(imgnames, masknames, config_dict, queue):
    TILE_KP = config_dict['use_tiling_non_max_supression']
    USE_MASKS = config_dict['use_masks']
    USE_CLAHE = config_dict['use_clahe']
    RADIAL_NON_MAX = config_dict['radial_non_max']
         
    if USE_CLAHE:
        clahe = cv2.createCLAHE(**config_dict['CLAHE_settings'])
    else: clahe = None
    
    if TILE_KP:
        tiling=[config_dict['tiling_non_max_tile_y'], config_dict['tiling_non_max_tile_x']]
    else: tiling = None
    
    if RADIAL_NON_MAX:
        RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
    else: RADIAL_NON_MAX_RADIUS = None
    
    detector = cv2.ORB_create(**config_dict['ORB_settings'])  
    
    print('Starting writer process...', flush=True)
    
    try:
        for i in range(len(imgnames)):
            if queue.empty(): print(Fore.RED+"Queue empty, reading is slow..."+Style.RESET_ALL)
            while queue.full():
                time.sleep(0.1)
                #print(Fore.GREEN+"Writer queue full, waiting..."+Style.RESET_ALL)
            if USE_MASKS:
                queue.put(preprocess_frame(imgnames[i], detector, masknames[i], clahe, 
                                           tiling, RADIAL_NON_MAX_RADIUS))
            else: queue.put(preprocess_frame(imgnames[i], detector, None, clahe, 
                                           tiling, RADIAL_NON_MAX_RADIUS))
    except KeyboardInterrupt:
        print ("Keyboard interrupt from me")
        passqq
    except:
        traceback.print_exc(file=sys.stdout)
    
    print("Finished pre-processing all images")


if __name__ == '__main__':
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
 #   TILEY=config_dict['tiling_non_max_tile_y']; 
#    TILEX=config_dict['tiling_non_max_tile_x']; 
#    TILE_KP = config_dict['use_tiling_non_max_supression']
    USE_MASKS = config_dict['use_masks']
#    USE_CLAHE = config_dict['use_clahe']
#    RADIAL_NON_MAX = config_dict['radial_non_max']
    RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
    image_folder = config_dict['image_folder']
    image_ext = config_dict['image_ext']
    init_imgs_indx = config_dict['init_image_indxs']
    img_step = config_dict['image_step']
    PAUSES = False
    PLOT_LANDMARKS = True
    FILTER_RP = False
    paused = False
    cue_to_exit = False
    
    images_full = sorted([f for f in glob.glob(path+image_folder+'/*') 
                     if re.match('^.*\.'+image_ext+'$', f, flags=re.IGNORECASE)])
    
    images = [images_full[init_imgs_indx[0]]]+images_full[init_imgs_indx[1]::img_step]
    
    assert images is not None, "ERROR: No images read"
    
    print(K,D)
            
    if USE_MASKS:
        masks_folder = config_dict['masks_folder']
        masks_ext = config_dict['masks_ext']
        masks_full = sorted([f for f in glob.glob(path+masks_folder+'/*') 
                        if re.match('^.*\.'+masks_ext+'$', f, flags=re.IGNORECASE)])
        masks = [masks_full[init_imgs_indx[0]]]+masks_full[init_imgs_indx[1]::img_step]
        assert len(masks)==len(images), "ERROR: Number of masks not equal to number of images"
    else:
        masks = None     
    
    #def onClick(event):
    #    print("Click")
    
    def onKey(event):
        global paused, cue_to_exit
        #print('you pressed', event.key, event.xdata, event.ydata)
        if event.key==" ":
            paused = not paused
        if event.key=="q":
            cue_to_exit = True
    
                
    mp.set_start_method('spawn')
    mpqueue = mp.Queue(5) # writer() writes to pqueue from _this_ process
    writer_p = mp.Process(target=writer, args=(images, masks, config_dict, mpqueue,))
    writer_p.daemon = True
    writer_p.start()        # Launch reader_proc() as a separate python process

    gr1, mask1, kp1_match_12 = mpqueue.get()
    gr2, mask2, kp2_pts = mpqueue.get()


    kp2_match_12, mask_klt, err = cv2.calcOpticalFlowPyrLK(gr1, gr2, kp1_match_12, None, **config_dict['KLT_settings'])
    print ("KLT tracked: ",np.sum(mask_klt) ," of total ",len(kp1_match_12),"keypoints")
    
    kp1_match_12 = kp1_match_12[mask_klt[:,0].astype(bool)]
    kp2_match_12 = kp2_match_12[mask_klt[:,0].astype(bool)]
    
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
    
    img12 = draw_point_tracks(kp1_match_12,gr2,kp2_match_12,mask_RP_12, False)
    
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
    kp1_match_12 = kp1_match_12[mask_RP_12[:,0].astype(bool)]
    kp2_match_12 = kp2_match_12[mask_RP_12[:,0].astype(bool)]
    
    kp1_match_12_ud = kp1_match_12_ud[mask_RP_12[:,0].astype(bool)]
    kp2_match_12_ud = kp2_match_12_ud[mask_RP_12[:,0].astype(bool)]
    
    landmarks_12, mask_tri_12 = triangulate(np.eye(4), T_1_2, kp1_match_12_ud, 
                                           kp2_match_12_ud, None)
    
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    fig2.subplots_adjust(0,0,1,1)
    plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
    ax2.set_aspect('equal')         # important!
    fig2.suptitle('Image 1 to 2 after triangulation')
    
    graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='r')
    set_axes_equal(ax2)
    ax2.view_init(0, -90)
    fig2.canvas.draw_idle(); plt.pause(0.01)
    graph.remove()
    
    if PLOT_LANDMARKS:
        graph = plot_3d_points(ax2, landmarks_12, linestyle="", marker=".", markersize=2, color='C0')
    
    kp2_match_12 = kp2_match_12[mask_tri_12[:,0].astype(bool)]
    
        
    cam_pose_0 = plot_pose3_on_axes(ax2,np.eye(4), axis_length=0.5)
    cam_pose = plot_pose3_on_axes(ax2, T_1_2, axis_length=1.0)
    
    cam_trail_pts = T_1_2[:3,[-1]].T
    cam_pose_trail = plot_3d_points(ax2, cam_trail_pts, linestyle="", color='g', marker=".", markersize=2)
    
    #fig2.canvas.mpl_connect('button_press_event', onClick)
    fig2.canvas.mpl_connect('key_press_event', onKey)
    
    plt.pause(.01)
    input("Press [enter] to continue.")
       
    kp2_pts = remove_redundant_newkps(kp2_pts, kp2_match_12, 5)
    
    print ("Points after redudancy check with current kps: ",len(kp2_pts))
    
    img12_newpts = draw_points(img12,kp2_pts[:,0,:], color=[255,255,0])
    fig1_image.set_data(img12_newpts)
    fig1.canvas.draw_idle(); plt.pause(0.01)
    if PAUSES: input("Press [enter] to continue.")
    
    print ("\n \n FRAME 2 COMPLETE \n \n")

    out = process_frame(*mpqueue.get(), gr2, kp2_match_12, 
                        kp2_pts, landmarks_12, T_1_2)
        
    print ("\n \n FRAME 3 COMPLETE \n \n")
    

    st = time.time()

    #for i in range(init_imgs_indx[1]+img_step*2,len(images),img_step):
    spinner = cycle(['|', '/', '-', '\\'])
    i = 0
    while not mpqueue.empty():
        while(paused):   
            print('\b'+next(spinner), end='', flush=True)
            plt.pause(0.1)
            if cue_to_exit: break
        if cue_to_exit: print("EXITING!!!"); raise SystemExit(0)
        
        ft = time.time()
        
        ppd = mpqueue.get()
        print(Fore.RED+"Time for ppd: "+str(time.time()-ft)+Style.RESET_ALL)
        
        out = process_frame(*ppd, *out)

        print(Fore.RED+"Time to process last frame: "+str(time.time()-st)+Style.RESET_ALL)
        print(Fore.RED+"Time in the function: "+str(time.time()-ft)+Style.RESET_ALL)

        st = time.time()

        plt.pause(0.001)
        print ("\n \n FRAME seq ", i ," COMPLETE \n \n")
        i+= 1
    
    writer_p.join()
    while(True):   
        print('\b'+next(spinner), end='', flush=True)
        plt.pause(0.5)
        if cue_to_exit: break
    plt.close(fig='all')