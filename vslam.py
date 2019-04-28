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
from helper_functions.frame import Frame
np.set_printoptions(precision=3,suppress=True)
import multiprocessing as mp
from colorama import Fore, Style
from itertools import cycle
import logging
import copyreg

from helper_functions.frame import Frame

'''
PROCESS FRAME
'''
global vslog
vslog = logging.getLogger('VSLAM')
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)8s  %(message)s')
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def writer(imgnames, masknames, config_dict, queue):
    '''
        
    '''
    
    #TILE_KP = config_dict['use_tiling_non_max_supression']
    USE_MASKS = config_dict['use_masks']
    USE_CLAHE = config_dict['use_clahe']
    ZERNIKE_settings = config_dict['ZERNIKE_settings']
    #RADIAL_NON_MAX = config_dict['radial_non_max']
    
    #detector = cv2.ORB_create(**config_dict['ORB_settings'])
    Frame.detector = MultiHarrisZernike(**ZERNIKE_settings)
    
    #settings_hash_string = str(hash(frozenset(a.items()))).replace('-','z')
    settings_string = ''.join(['_%s' % ZERNIKE_settings[k] for k in sorted(ZERNIKE_settings.keys())])
    local_temp_dir = os.path.dirname(os.path.abspath(__file__))+'/temp_data'
    img_folder_name = os.path.dirname(imgnames[0]).replace('/','_')[1:]
    
    temp_obj_folder = local_temp_dir+'/'+img_folder_name+settings_string
    os.makedirs(temp_obj_folder, exist_ok=True)
    
    vslog.info("K: \t"+str(Frame.K).replace('\n','\n\t\t'))
    vslog.info("D: \t"+str(Frame.D))
        
    if USE_CLAHE:
        Frame.clahe_obj = cv2.createCLAHE(**config_dict['CLAHE_settings'])
    
    Frame.is_config_set = True
 
    '''
    if TILE_KP:
        tiling=[config_dict['tiling_non_max_tile_y'], config_dict['tiling_non_max_tile_x']]
    else: tiling = None
    
    if RADIAL_NON_MAX:
        RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
    else: RADIAL_NON_MAX_RADIUS = None
    '''
    
    vslog.info('Starting writer process...')
    
    try:
        for i in range(len(imgnames)):
            frame_obj_filename = temp_obj_folder+'/'+imgnames[i].split('/')[-1]+'.pkl'
            if os.path.isfile(frame_obj_filename):
                vslog.debug(Fore.GREEN+"File exisits, reusing ..."+Style.RESET_ALL)
                fr = Frame.load_frame(frame_obj_filename)
                Frame.last_id += 1                
                queue.put(fr)
            else:                
                if queue.empty(): vslog.debug(Fore.RED+"Queue empty, reading is slow..."+Style.RESET_ALL)
                while queue.full():
                    time.sleep(0.01)
                    #print(Fore.GREEN+"Writer queue full, waiting..."+Style.RESET_ALL)
                if USE_MASKS:
                    fr = Frame(imgnames[i],mask_name=masknames[i])
                else: 
                    fr = Frame(imgnames[i])
                    #gr, mask, kp, des = ppoutput
                queue.put(fr)
                Frame.save_frame(fr, frame_obj_filename)
            
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
                        default='config/kitti_config.yaml') #go_pro_icebergs_config.yaml
    args = parser.parse_args()
     
    # Inputs, images and camera info
        
    config_dict = yaml.load(open(args.config))
    CHESSBOARD = config_dict['chessboard']
    USE_MASKS = config_dict['use_masks']
    #RADIAL_NON_MAX_RADIUS = config_dict['radial_non_max_radius']
    image_ext = config_dict['image_ext']
    init_imgs_indx = config_dict['init_image_indxs']
    img_step = config_dict['image_step']
    PLOT_LANDMARKS = config_dict['plot_landmarks']
    findEssential_set = config_dict['findEssential_settings']
    
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
    FILTER_RP = False
    paused = False
    cue_to_exit = False
    
    images_full = sorted([f for f in glob.glob(image_folder+'/*') 
                     if re.match('^.*\.'+image_ext+'$', f, flags=re.IGNORECASE)])
    
    images = [images_full[init_imgs_indx[0]]]+images_full[init_imgs_indx[1]::img_step]
    
    assert images is not None, "ERROR: No images read"
    
            
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
    
    # Configure settings for Frame object
    Frame.K = np.array(config_dict['K'])
    Frame.D = np.array(config_dict['D'])
    Frame.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    Frame.config_dict = config_dict

    # Launch the pre-processing thread                
    mp.set_start_method('spawn',force=True)
    mpqueue = mp.Queue(5) # writer() writes to pqueue from _this_ process
    writer_p = mp.Process(target=writer, args=(images, masks, config_dict, mpqueue))
    writer_p.daemon = True
    writer_p.start()
    
    # Process first 2 frames
    fr1 = mpqueue.get()
    #gr1, mask1, kp1, des1 = fr1.gr,fr1.mask,fr1.kp,fr1.des
    fr2 = mpqueue.get()
    #gr2, mask2, kp2, des2 = fr2.gr,fr2.mask,fr2.kp,fr2.des
    
    # Show image
    Frame.initialize_figures(window_xadj, window_yadj)
    Frame.fig1.canvas.mpl_connect('key_press_event', onKey)
    Frame.fig2.canvas.mpl_connect('key_press_event', onKey)
    Frame.initialize_VSLAM(fr1, fr2)

    Frame.frlog.info(Fore.GREEN+"\tFRAME 1 COMPLETE\n"+Style.RESET_ALL)
    
    fr_prev = fr2

    st = time.time()
    #for i in range(init_imgs_indx[1]+img_step*2,len(images),img_step):
    spinner = cycle(['|', '/', '-', '\\'])
    i = 4
    while True:
        if not mpqueue.empty():
            while(paused):   
                print('\b'+next(spinner), end='', flush=True)
                #plt.pause(0.1)
                Frame.fig1.canvas.start_event_loop(0.001)
                Frame.fig2.canvas.start_event_loop(0.001)
                time.sleep(0.1)
                if cue_to_exit: break
            if cue_to_exit: vslog.info("EXITING!!!"); raise SystemExit(0)
            
            ft = time.time()
            
            fr_curr = mpqueue.get()
            Frame.frlog.debug("Frame id:"+str(fr_curr.frame_id))
            Frame.frlog.debug(Fore.RED+"Time for current frame: "+str(time.time()-ft)+Style.RESET_ALL)

            Frame.process_keyframe(fr_prev, fr_curr)
    
            Frame.frlog.debug(Fore.RED+"Time to process last frame: {:.4f}".format(time.time()-st)+Style.RESET_ALL)
            Frame.frlog.debug(Fore.RED+"Time in the function: {:.4f}".format(time.time()-ft)+Style.RESET_ALL)
            Frame.frlog.info(Fore.GREEN+"\tFRAME seq {} COMPLETE \n".format(fr_curr.frame_id)+Style.RESET_ALL)
            
            st = time.time()
            fr_prev = fr_curr
            #plt.pause(0.001)
            
            i+= 1
        else: time.sleep(0.2)            
    
    writer_p.join()
    while(True):   
        print('\b'+next(spinner), end='', flush=True)
        #plt.pause(0.5)
        if cue_to_exit: break
    plt.close(fig='all')
