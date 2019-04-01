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
from zernike import MultiHarrisZernike
np.set_printoptions(precision=3,suppress=True)
import multiprocessing as mp
from colorama import Fore, Style
from itertools import cycle
import logging
'''
PROCESS FRAME
'''
global vslog
vslog = logging.getLogger('VSLAM')
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)8s  %(message)s')
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

#formatter = logging.Formatter("%(filename)s: %(levelname)8s %(message)s")
#console = logging.StreamHandler()
#console.setLevel(logging.DEBUG)
#console.setFormatter(formatter)
#vslog.addHandler(console)


#vslog.setLevel(logging.DEBUG)
frame_no = 3

# j = current, i = previous frame
def process_frame(gr_j, mask_j, kp_j, des_j, gr_i, kp_i, des_i, 
                  kp_i_cand, des_i_cand, lm_i, T_i):
    global frame_no, cam_pose_trail, cam_trail_pts, cam_pose, new_lm_graph, vslog
    vslog = logging.getLogger('VSLAM')
    vslog.info("len of lm_i: {}".format(len(lm_i)))
    vslog.info("len of kp_i: {}".format(len(kp_i)))

    time_start = time.time()
        
    prop_out = propogate_kp(matcher, kp_i, des_i, kp_i_cand, des_i_cand, lm_i, 
                            kp_j, des_j,threshold=config_dict['lowe_ratio_test_threshold'])
    kpil_match, kpic_match, kpjl_match, desjl_match, kpjn_match, desjn_match, kp_j_cand, des_j_cand, lm_if = prop_out

    '''
    if len(kp_prev_cand)>0:
        kp_curr_cand_matchpc, mask_cand_klt, _ = cv2.calcOpticalFlowPyrLK(gr_prev, gr_curr, 
                                                            kp_prev_cand, None, **config_dict['KLT_settings'])
        print ("KLT Candidates Tracked: ",np.sum(mask_cand_klt)," of total ",len(kp_prev_cand),"keypoints")
        kp_prev_cand = kp_prev_cand[mask_cand_klt[:,0].astype(bool)]
        kp_curr_cand_matchpc = kp_curr_cand_matchpc[mask_cand_klt[:,0].astype(bool)]
    else: 
        kp_curr_cand_matchpc = np.zeros([0,1,2])
        print ("No New KLT Candidates Tracked. ")
    '''
    
    vslog.debug("Time elapsed in matcher flow: {:.4f}".format(time.time()-time_start)); 
    time_start = time.time()

    kp_i_all = np.vstack((kpil_match, kpic_match))
    kp_j_all = np.vstack((kpjl_match, kpjn_match))
    
    kp_i_ud = cv2.undistortPoints(np.expand_dims(kp_i_all,1),K,D)[:,0,:]
    kp_j_ud = cv2.undistortPoints(np.expand_dims(kp_j_all,1),K,D)[:,0,:]

    vslog.debug("Time elapsed in undistort: {:.4f}".format(time.time()-time_start))
    time_start = time.time()

    E, mask_e_all = cv2.findEssentialMat(kp_i_ud, kp_j_ud, focal=1.0, pp=(0., 0.), 
                                   method=cv2.RANSAC, **config_dict['findEssential_settings'])
    essen_mat_pts = np.sum(mask_e_all)  
    vslog.debug("Time elapsed in find E: {:.4f}".format(time.time()-time_start)) 
    time_start = time.time()
    
    vslog.debug("Essential matrix: used {} of total {} matches".format(essen_mat_pts,len(kp_j_all)))
    
    if FILTER_RP:
        # Recover Pose filtering is breaking under certain conditions. Leave out for now.
        _, _, _, mask_RP_all = cv2.recoverPose(E, kp_i_ud, kp_j_ud, mask=mask_e_all)
        vslog.info("Recover pose: used {} of total {} matches".format(np.sum(mask_RP_all),essen_mat_pts))
    else: 
        mask_RP_all = mask_e_all
    
    # Split the combined mask to lm features and candidates
    mask_RP_lm = mask_RP_all[:len(kpil_match)]
    mask_RP_cand = mask_RP_all[-len(kpic_match):]
    
    # Display translucent mask on image.
    if mask_j is not None:
        gr_j_masked = cv2.addWeighted(mask_j, 0.2, gr_j, 1 - 0.2, 0)
    else: gr_j_masked = gr_j
    
    img_track_lm = draw_point_tracks(kpil_match,gr_j_masked,
                                       kpjl_match,mask_RP_lm, False)
    img_track_all = draw_point_tracks(kpic_match,img_track_lm,
                                       kpjn_match,mask_RP_cand, False, color=[255,255,0])

    vslog.debug("Time elapsed in drawing tracks: {:.4f}".format(time.time()-time_start))
    time_start = time.time()

    vslog.debug('shape of mask_RP_lm: {}, shape of lm_if'.format(mask_RP_lm.shape,lm_i.shape))

    lm_if_up = lm_if[mask_RP_lm[:,0].astype(bool)]

    vslog.info("LM prev(i) was: {}. LM(i) updated after Ess mat filter is: {}".format(len(lm_if),len(lm_if_up)))
    kpjl_match = kpjl_match[mask_RP_lm[:,0].astype(bool)]
    desjl_match = desjl_match[mask_RP_lm[:,0].astype(bool)]
           
    success, T_j, inliers = T_from_PNP(lm_if_up, kpjl_match, K, D)
    vslog.debug("Time elapsed in PNP: {:.4f}".format(time.time()-time_start))
    time_start = time.time()

    if not success:
        vslog.critical("PNP failed in frame {}. Exiting...".format(frame_no+1))
        exit()
        
    vslog.info("PNP inliers: {}  of {}".format(len(inliers),len(lm_if_up)))

                           
    kpic_ud = kp_i_ud[-len(kpic_match):]
    kpjn_ud = kp_j_ud[-len(kpjn_match):]
    
    kpic_match = kpic_match[mask_RP_cand[:,0].astype(bool)]    
    kpjn_match = kpjn_match[mask_RP_cand[:,0].astype(bool)]
    desjn_match = desjn_match[mask_RP_cand[:,0].astype(bool)]   
    kpic_ud = kpic_ud[mask_RP_cand[:,0].astype(bool)]
    kpjn_ud = kpjn_ud[mask_RP_cand[:,0].astype(bool)]
       
    lm_j, mask_tri = triangulate(T_i, T_j, kpic_ud, kpjn_ud, None)
    vslog.debug("Time elapsed in triangulate: {:.4f}".format(time.time()-time_start)) 
    time_start = time.time()

    vslog.debug("Points after rejection in triangulation: {} out of length: {}".format(np.sum(mask_tri),len(mask_tri)))
    
    #if len(kp_prev_cand)>0:
    img_rej_pts = draw_point_tracks(kpic_match, img_track_all, kpjn_match, 
                                    1-mask_tri, False, color=[255,0,0])
    #else: img_rej_pts = img_track_all
    vslog.debug("Time elapsed in draw pt tracks: {:.4f} ".format(time.time()-time_start)) 
    time_start = time.time()

    kpjn_match = kpjn_match[mask_tri[:,0].astype(bool)]
    desjn_match = desjn_match[mask_tri[:,0].astype(bool)]
    
    try:    
        new_lm_graph.remove()
    except NameError:
        pass
    new_lm_graph = plot_3d_points(ax2, lm_j, linestyle="", color='r', marker=".", markersize=2)
    plot_pose3_on_axes(ax2, T_j, axis_length=2.0, center_plot=True, line_obj_list=cam_pose)
    
    cam_trail_pts = np.append(cam_trail_pts,T_j[:3,[-1]].T,axis=0)
    cam_pose_trail = plot_3d_points(ax2,cam_trail_pts , line_obj=cam_pose_trail, linestyle="", color='g', marker=".", markersize=2)
    fig2.canvas.draw_idle(); #plt.pause(0.01)
    if PAUSES: input("Press [enter] to continue.")

    # Remove pose lines and landmarks to speed plotting
    if PLOT_LANDMARKS:
        graph_newlm = plot_3d_points(ax2, lm_j, linestyle="", color='C0', marker=".", markersize=2)    
        fig2.canvas.draw_idle(); #plt.pause(0.01)
    
    lm_j_up = np.concatenate((lm_if_up,lm_j))
    kpjl = np.concatenate((kpjl_match,kpjn_match))
    desjl = np.concatenate((desjl_match,desjn_match))
    
    vslog.debug("Candidate points: {}".format(len(kp_j_cand)))
    img_cand_pts = draw_points(img_rej_pts,kp_j_cand, color=[255,255,0])
    fig1_image.set_data(img_cand_pts)
    fig1.canvas.draw_idle(); plt.pause(0.05)
    
    frame_no += 1
        
    vslog.debug("FRAME seq {} COMPLETE".format(str(frame_no)))

    return gr_j, kpjl, desjl, kp_j_cand, des_j_cand, lm_j_up, T_j

def preprocess_frame(image_name, detector, mask_name=None, clahe_obj=None, tiling=None, rnm_radius=None):
    vslog = logging.getLogger('VSLAM')
    vslog.debug("Pre-processing image: "+image_name)

    pt = time.time()
    img = cv2.imread(image_name)
    gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if mask_name is not None:
        mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
    else: mask= None

    if clahe_obj is not None: gr = clahe_obj.apply(gr)
    
    kp,des = detector.detectAndCompute(gr,mask)
    pbf = "New feature candidates detected: "+str(len(kp))
    
    if tiling is not None:
        kp = tiled_features(kp, gr.shape, *tiling)
        pbf += " > tiling supression: "+str(len(kp))
    
    if rnm_radius is not None:
        kp = radial_non_max(kp,rnm_radius)
        pbf += " > radial supression: "+str(len(kp))
        
    kp_pts = cv2.KeyPoint_convert(kp)
    
    vslog.debug(pbf)
    vslog.debug("Image Pre-processing time is {:.4f}".format(time.time()-pt))
    return gr, mask, kp_pts, des

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
    
    #detector = cv2.ORB_create(**config_dict['ORB_settings'])
    detector = MultiHarrisZernike(**config_dict['ZERNIKE_settings'])
    
    vslog.info('Starting writer process...')
    
    try:
        for i in range(len(imgnames)):
            if queue.empty(): vslog.debug(Fore.RED+"Queue empty, reading is slow..."+Style.RESET_ALL)
            while queue.full():
                time.sleep(0.01)
                #print(Fore.GREEN+"Writer queue full, waiting..."+Style.RESET_ALL)
            if USE_MASKS:
                queue.put(preprocess_frame(imgnames[i], detector, masknames[i], clahe, 
                                           tiling, RADIAL_NON_MAX_RADIUS))
            else: 
                ppoutput = preprocess_frame(imgnames[i], detector, None, clahe, 
                                            tiling, RADIAL_NON_MAX_RADIUS)
                gr, mask, kp, des = ppoutput
                queue.put(ppoutput)
    except KeyboardInterrupt:
        vslog.critical("Keyboard interrupt from me")
        pass
    except:
        traceback.print_exc(file=sys.stdout)
    
    vslog.info("Finished pre-processing all images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the simple VSLAM pipeline')
    parser.add_argument('-c', '--config', help='location of config file in yaml format',
                        default='config/kitti_config.yaml') #go_pro_icebergs_config.yaml
    args = parser.parse_args()
     
    # Inputs, images and camera info
        
    config_dict = yaml.load(open(args.config))
    K = np.array(config_dict['K'])
    D = np.array(config_dict['D'])
    CHESSBOARD = config_dict['chessboard']
    USE_MASKS = config_dict['use_masks']
    #RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
    image_ext = config_dict['image_ext']
    init_imgs_indx = config_dict['init_image_indxs']
    img_step = config_dict['image_step']
    PLOT_LANDMARKS = config_dict['plot_landmarks']
    
    if sys.platform == 'darwin':
        image_folder = config_dict['osx_image_folder']
        if USE_MASKS: masks_folder = config_dict['osx_masks_folder']

        window_xadj = 0
        window_yadj = 45
    else:
        image_folder = config_dict['linux_image_folder']
        if USE_MASKS: masks_folder = config_dict['linux_masks_folder']
        window_xadj = 65
        window_yadj = 430


    PAUSES = False
    FILTER_RP = True
    paused = False
    cue_to_exit = False
    
    images_full = sorted([f for f in glob.glob(image_folder+'/*') 
                     if re.match('^.*\.'+image_ext+'$', f, flags=re.IGNORECASE)])
    
    images = [images_full[init_imgs_indx[0]]]+images_full[init_imgs_indx[1]::img_step]
    
    assert images is not None, "ERROR: No images read"
    
    vslog.info("K: \t"+str(K).replace('\n','\n\t\t'))
    vslog.info("D: \t"+str(D))

            
    if USE_MASKS:
        masks_ext = config_dict['masks_ext']
        masks_full = sorted([f for f in glob.glob(masks_folder+'/*') 
                        if re.match('^.*\.'+masks_ext+'$', f, flags=re.IGNORECASE)])
        masks = [masks_full[init_imgs_indx[0]]]+masks_full[init_imgs_indx[1]::img_step]
        assert len(masks)==len(images), "ERROR: Number of masks not equal to number of images"
    else:
        masks = None     
    
    def onKey(event):
        global paused, cue_to_exit
        #print('you pressed', event.key, event.xdata, event.ydata)
        if event.key==" ":
            paused = not paused
        if event.key=="q":
            cue_to_exit = True
    
    # Launch the pre-processing thread                
    mp.set_start_method('spawn')
    mpqueue = mp.Queue(5) # writer() writes to pqueue from _this_ process
    writer_p = mp.Process(target=writer, args=(images, masks, config_dict, mpqueue,))
    writer_p.daemon = True
    writer_p.start()        # Launch reader_proc() as a separate python process
    
    # Process first 2 frames
    gr1, mask1, kp1, des1 = mpqueue.get()
    gr2, mask2, kp2, des2 = mpqueue.get()
    #vslog.debug(' kp1: '+str(kp1[0,:])+' kp2: '+str(kp2[0,:]))
    vslog.debug('Length of kp1: {}  Length of kp2: {}'.format(len(kp1),len(kp2)))

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_12 = knn_match_and_lowe_ratio_filter(matcher, des1, des2, 
                                                 threshold=config_dict['lowe_ratio_test_threshold'])
    vslog.debug('Length of matches: '+str(len(matches_12)))
    
    track_output = track_kp_array(kp1, des1, kp2, des2, matches_12)
    kp1_match_12, des1_m, kp2_match_12, des2_m, kp2_cand_pts, des2_cand = track_output
    
    #kp1_match_12 = np.expand_dims(kp1_matched,1)
    #kp2_match_12 = np.expand_dims(kp2_matched,1)
    
    kp1_match_12_ud = cv2.undistortPoints(np.expand_dims(kp1_match_12,1),K,D)[:,0,:]
    kp2_match_12_ud = cv2.undistortPoints(np.expand_dims(kp2_match_12,1),K,D)[:,0,:]
    
    E_12, mask_e_12 = cv2.findEssentialMat(kp1_match_12_ud, kp2_match_12_ud, focal=1.0, pp=(0., 0.), 
                                           method=cv2.RANSAC, **config_dict['findEssential_settings'])
    vslog.info("Essential matrix: used {} of total {} matches".format(np.sum(mask_e_12),len(kp1_match_12),))
    essen_mat_pts = np.sum(mask_e_12)
    points, R_21, t_21, mask_RP_12 = cv2.recoverPose(E_12, kp1_match_12_ud, kp2_match_12_ud,mask=mask_e_12)
    vslog.info("Recover pose used {} of total matches in Essential matrix".format(np.sum(mask_RP_12),essen_mat_pts))
    T_2_1 = compose_T(R_21,t_21)
    T_1_2 = T_inv(T_2_1)
    vslog.info("R:\t"+str(R_21).replace('\n','\n\t\t'))
    vslog.info("t:\t"+str(t_21.T))
        
    img12 = draw_point_tracks(kp1_match_12,gr2,kp2_match_12,mask_RP_12, True)
    
    fig1 = plt.figure(1)
    plt.get_current_fig_manager().window.setGeometry(window_xadj,window_yadj,640,338)
    
    fig1_image = plt.imshow(img12)
    plt.title('Image 1 to 2 matches')
    plt.axis("off")
    fig1.subplots_adjust(0,0,1,1)
    plt.draw()
    plt.pause(0.001)
    fig1.canvas.mpl_connect('key_press_event', onKey)
    input("Press [enter] to continue.\n")
    
    # Trim the tracked key pts
    kp1_match_12 = kp1_match_12[mask_RP_12[:,0].astype(bool)]
    kp2_match_12 = kp2_match_12[mask_RP_12[:,0].astype(bool)]
    des2_m = des2_m[mask_RP_12[:,0].astype(bool)]
    
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
    des2_m = des2_m[mask_tri_12[:,0].astype(bool)]
            
    cam_pose_0 = plot_pose3_on_axes(ax2,np.eye(4), axis_length=0.5)
    cam_pose = plot_pose3_on_axes(ax2, T_1_2, axis_length=1.0)
    
    cam_trail_pts = T_1_2[:3,[-1]].T
    cam_pose_trail = plot_3d_points(ax2, cam_trail_pts, linestyle="", color='g', marker=".", markersize=2)
    
    #fig2.canvas.mpl_connect('button_press_event', onClick)
    fig2.canvas.mpl_connect('key_press_event', onKey)
    
    plt.pause(.01)
    input("Press [enter] to continue.\n")
       
    #kp2_pts = remove_redundant_newkps(kp2_pts, kp2_match_12, 5)
    
    vslog.debug("Length of candidate pts: {}".format(len(kp2_cand_pts)))
    #kp2_cand_pts = kp2_cand
    
    img12_newpts = draw_points(img12,kp2_cand_pts, color=[255,255,0])
    fig1_image.set_data(img12_newpts)
    fig1.canvas.draw_idle(); plt.pause(0.01)
    if PAUSES: input("Press [enter] to continue.")
    
    vslog.info(Fore.GREEN+"\tFRAME 2 COMPLETE\n"+Style.RESET_ALL)

    out = process_frame(*mpqueue.get(), gr2, kp2_match_12, des2_m,
                        kp2_cand_pts, des2_cand, landmarks_12, T_1_2)
        
    vslog.info(Fore.GREEN+"\tFRAME 3 COMPLETE\n"+Style.RESET_ALL)
    

    st = time.time()

    #for i in range(init_imgs_indx[1]+img_step*2,len(images),img_step):
    spinner = cycle(['|', '/', '-', '\\'])
    i = 0
    while True:
        if not mpqueue.empty():
            while(paused):   
                print('\b'+next(spinner), end='', flush=True)
                plt.pause(0.1)
                if cue_to_exit: break
            if cue_to_exit: vslog.info("EXITING!!!"); raise SystemExit(0)
            
            ft = time.time()
            
            ppd = mpqueue.get()
            vslog.debug(Fore.RED+"Time for ppd: "+str(time.time()-ft)+Style.RESET_ALL)
            
            out = process_frame(*ppd, *out)
    
            vslog.debug(Fore.RED+"Time to process last frame: {:.4f}".format(time.time()-st)+Style.RESET_ALL)
            vslog.debug(Fore.RED+"Time in the function: {:.4f}".format(time.time()-ft)+Style.RESET_ALL)
    
            st = time.time()
    
            plt.pause(0.001)
            vslog.info(Fore.GREEN+"\tFRAME seq {} COMPLETE \n".format(i)+Style.RESET_ALL)
            i+= 1
        else: time.sleep(0.2)            
    
    writer_p.join()
    while(True):   
        print('\b'+next(spinner), end='', flush=True)
        plt.pause(0.5)
        if cue_to_exit: break
    plt.close(fig='all')