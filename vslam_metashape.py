#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os, sys
from vslam_helper import *
from utils.ssc import *
import yaml
import glob
import re
import argparse
import traceback
from zernike.zernike import MultiHarrisZernike
from helper_functions.frame_metashape import Frame_metashape
np.set_printoptions(precision=5,suppress=True)
import multiprocessing as mp
from colorama import Fore, Back, Style
from itertools import cycle
import logging
import copyreg
import queue
import pickle

from GTSAM_helper import iSAM2Wrapper


global vslog
vslog = logging.getLogger('VSLAM')
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)8s  %(message)s')
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def writer(config_dict, queue):
    '''
    This funtion creates Frame objects which read images from the disk, detects features 
    and feature descriptors using settings in the config file. It then puts the object into
    a multi-process queue.
    This function is designed to run in a separate heap and thus takes everything it needs
    in form of parameteres and doesn't rely on global variables.    
    '''        
    
    if sys.platform == 'darwin':
        Frame_metashape.image_folder = config_dict['osx_image_folder']
        data_folder = config_dict['osx_data_folder']
    else:
        Frame_metashape.image_folder = config_dict['linux_image_folder']
        data_folder = config_dict['linux_data_folder']
        
    with open(data_folder+config_dict['data_pkl_file'], 'rb') as input:
        frames_full, points, K, D = pickle.load(input)
    
    init_imgs_indx = config_dict['init_image_indxs']
    img_step = config_dict['image_step']
    
    frames = [frames_full[init_imgs_indx[0]]]+frames_full[init_imgs_indx[1]::img_step]
    
    Frame_metashape.K = K
    Frame_metashape.D = D
    Frame_metashape.config_dict = config_dict
    
    vslog.info("K: \t"+str(Frame_metashape.K).replace('\n','\n\t\t'))
    vslog.info("D: \t"+str(Frame_metashape.D))
            
    Frame_metashape.is_config_set = True
    
    vslog.info('Starting writer process...')
    
    try:
        for i in range(len(frames)):
            if queue.empty(): vslog.debug(Fore.RED+"Queue empty, reading is slow..."+Style.RESET_ALL)
            while queue.full():
                time.sleep(0.01)
                #print(Fore.GREEN+"Writer queue full, waiting..."+Style.RESET_ALL)
            fr = Frame_metashape(frames[i])
            queue.put(fr)
             
    except KeyboardInterrupt:
        vslog.critical("Keyboard interrupt from me")
        pass
    except:
        traceback.print_exc(file=sys.stdout)

    vslog.info("Finished pre-processing all images")


if __name__ == '__main__':    
    # passing arguments from the terminal
    parser = argparse.ArgumentParser(description='This is the simple VSLAM pipeline')
    parser.add_argument('-c', '--config', help='location of config file in yaml format',
                        default='config/go_pro_Lars_metashape.yaml') 
    args = parser.parse_args()
     
    # Inputs, images and camera info
    config_dict = yaml.load(open(args.config))
    image_ext = config_dict['image_ext']
    init_imgs_indx = config_dict['init_image_indxs']
    img_step = config_dict['image_step']
    PLOT_LANDMARKS = config_dict['plot_landmarks']
    findEssential_set = config_dict['findEssential_settings']
    PAUSES = config_dict['pause_every_iteration']
    USE_GTSAM = config_dict['use_gtsam']

    
    if sys.platform == 'darwin':
        window_xadj = 0
        window_yadj = 45
    else:
        window_xadj = 65
        window_yadj = 430

    FILTER_RP = False
    paused = False
    cue_to_exit = False

    def onKey(event):
        global paused, cue_to_exit, PAUSES
        #print('you pressed', event.key, event.xdata, event.ydata)
        if event.key==" ":
            paused = not paused
        if event.key=="q":
            cue_to_exit = True
        if event.key=="p":
            PAUSES = not PAUSES
            paused = not paused
    
    # Configure settings for Frame object
    Frame_metashape.config_dict = config_dict

    # Launch the pre-processing thread                
    mp.set_start_method('spawn',force=True)
    mpqueue = mp.Queue(5) # writer() writes to pqueue from _this_ process
    writer_p = mp.Process(target=writer, args=(config_dict, mpqueue))
    writer_p.daemon = True
    writer_p.start()
    
    # Process first 2 frames
    fr1 = mpqueue.get()
    #gr1, mask1, kp1, des1 = fr1.gr,fr1.mask,fr1.kp,fr1.des
    fr2 = mpqueue.get()
    #gr2, mask2, kp2, des2 = fr2.gr,fr2.mask,fr2.kp,fr2.des
    
    frame_queue = queue.Queue(maxsize=5)
    frame_queue.put(fr1)
    frame_queue.put(fr2)
    
    # Show image
    Frame_metashape.initialize_figures(window_xadj, window_yadj)
    Frame_metashape.fig1.canvas.mpl_connect('key_press_event', onKey)
    Frame_metashape.fig2.canvas.mpl_connect('key_press_event', onKey)
    Frame_metashape.initialize_VSLAM(fr1, fr2)
    
    factor_graph = iSAM2Wrapper(pose0=np.eye(4), K=np.eye(3), **config_dict['iSAM2_settings'])    
    factor_graph.add_PoseEstimate(fr2.frame_id, fr2.T_pnp)      
    
    plt.pause(0.001)
    
    Frame_metashape.frlog.info(Fore.GREEN + Back.BLUE +"\tFRAME 1 COMPLETE\n"+Style.RESET_ALL)
    
    fr_prev = fr2

    st = time.time()
    #for i in range(init_imgs_indx[1]+img_step*2,len(images),img_step):
    spinner = cycle(['|', '/', '-', '\\'])
    i = 4
    flag = False
    while True:
        if not mpqueue.empty():
            if cue_to_exit: 
                vslog.info("EXITING!!!")
                plt.close('all')
                raise SystemExit(0)
            
            ft = time.time()
            
            fr_curr = mpqueue.get()
            
            #if frame_queue.full(): frame_queue.get()
            #frame_queue.put(fr_curr)
            
            Frame_metashape.frlog.debug("Frame id:"+str(fr_curr.frame_id))
            Frame_metashape.frlog.debug(Fore.RED+"Time for current frame: "+str(time.time()-ft)+Style.RESET_ALL)
            ft = time.time()
            Frame_metashape.process_keyframe_PNP(fr_prev, fr_curr)
            Frame_metashape.frlog.debug("Time elapsed in process_keyframe: {:.4f}".format(time.time()-ft))

            ft = time.time()
            
            if USE_GTSAM:
                factor_graph.add_keyframe_factors(fr_curr)
                            
                factor_graph.update(1)
                
                fr_curr.T_gtsam = factor_graph.get_curr_Pose_Estimate(fr_curr.frame_id)  
                fr_curr.T_pnp = fr_curr.T_gtsam 

                #current_estimate = factor_graph.get_Estimate()
                corr_landmarks, gtsam_lm_ids = factor_graph.get_landmark_estimates()
                
                mean_correction = np.mean(np.linalg.norm(corr_landmarks - Frame_metashape.landmarks[gtsam_lm_ids],axis=1))
                max_correction = np.max(np.linalg.norm(corr_landmarks - Frame_metashape.landmarks[gtsam_lm_ids],axis=1))
                Frame_metashape.frlog.info("GTAM Landmark correction: Mean: {:.3f} Max: {:.3f}".format(mean_correction,max_correction))
                
                Frame_metashape.landmarks[gtsam_lm_ids] = corr_landmarks
                trans_correction = np.linalg.norm(fr_curr.T_gtsam[:3,-1]-fr_curr.T_gtsam[:3,-1])
                rot_correction = rotation_distance(fr_curr.T_gtsam[:3,:3], fr_curr.T_pnp[:3,:3])
                Frame_metashape.frlog.info("GTSAM correction: Trans: {:.5f} rot angle: {:.4f} deg".format(trans_correction,rot_correction))
                Frame_metashape.frlog.info("Time elapsed in iSAM optimization: {:.4f}".format(time.time()-ft))
            
            Frame_metashape.process_keyframe_triangulation(fr_prev, fr_curr)
            
            Frame_metashape.frlog.debug(Fore.RED+"Time to process last frame: {:.4f}".format(time.time()-st)+Style.RESET_ALL)
            Frame_metashape.frlog.debug(Fore.RED+"Time in the function: {:.4f}".format(time.time()-ft)+Style.RESET_ALL)
            Frame_metashape.frlog.info(Fore.GREEN + Back.BLUE + "\tFRAME seq {} COMPLETE \n".format(fr_curr.frame_id)+Style.RESET_ALL)
            
            
            
            st = time.time()
            
            fr_prev=fr_curr
            
            if PAUSES: paused=True

            while(paused):   
                print('\b'+next(spinner), end='', flush=True)
                #plt.pause(0.1)
                Frame_metashape.fig1.canvas.start_event_loop(0.001)
                Frame_metashape.fig2.canvas.start_event_loop(0.001)

                if cue_to_exit: 
                    flag = True
                    break
                time.sleep(0.2)
            
            i+= 1
        else: time.sleep(0.2)            
    
    writer_p.join()
    while(True):   
        print('\b'+next(spinner), end='', flush=True)
        plt.pause(0.5)
        if cue_to_exit: break
    plt.close(fig='all')
