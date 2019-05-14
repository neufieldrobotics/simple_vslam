#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this script on a Metashape project with has a sparse point cloud to export a 
pickle file which contains a tuple of 4 containing:
    1. frames: A list of dictionaries containing:
        a. 'name' - string of name of image file without extension.
        b. 'kp' - Nx2 float32 numpy array of keypoints in each image
        c. 'track_ids' - Nx1 int numpy array of 'track_ids' for keypoint
            tracks (chunk.point_cloud.tracks) in Python API represent all matching 
            points found, even if there are no 3D locations estimated for them. 
            Whereas each element from the point_cloud.points is linking to one 
            of the point_cloud.tracks, so len(tracks) >= len(points).
        d. 'transform' - 4 x 4 numpy array of the calculated camera pose
    2. points_dict - A dictionary which uses 'track_ids' as a key contains calulated
                     3D coordinates for each point (landmark)
    3. K - 3 x 3 numpy array of calculated Camera matrix (opencv format)
    4. D - 1 x N (4 or 5) numpy array of calculated Distortion coefficients

@author: vik748
"""

import Metashape
import numpy as np
import yaml
import pickle
import xml.etree.ElementTree as ET

doc = Metashape.app.document
doc.open('/data/Lars/Project_files/Lars2_081018_clahe_40000_gap_opposite_scripting.psx')
chunk = doc.chunk
point_cloud = chunk.point_cloud
points = point_cloud.points
projections = point_cloud.projections

npoints = len(points)

'''
selected_photos = list()
for photo in chunk.cameras:
	if photo.selected:
		selected_photos.append(photo)
'''

selected_photos = chunk.cameras #[200:205]

'''
for photo in selected_photos:
	
	point_index = 0
	for proj in projections[photo]:
		track_id = proj.track_id
		while point_index < npoints and points[point_index].track_id < track_id:
			point_index += 1
		if point_index < npoints and points[point_index].track_id == track_id:
			if not points[point_index].valid: 
				continue	
			else:
				points[point_index].selected = True
'''

Metashape.app.update()

print("Highlighted points for the following cameras: ")
print(selected_photos)

frames = []
for cam in selected_photos:
    if cam.enabled:
        kp_list=[]
        track_id_list = []
        for pj in projections[cam]:
            kp_list.append([pj.coord.x, pj.coord.y])
            track_id_list.append(pj.track_id)
        frame_data = {}
        frame_data['name'] = cam.label
        frame_data['kp'] = np.array(kp_list,dtype='float32')
        frame_data['track_ids'] = np.array(track_id_list)
        frame_data['transform'] = np.array(cam.transform).reshape(4,4)
        
        frames.append(frame_data)
    
#with open('data.yml', 'w') as outfile:
#    yaml.dump(frames, outfile, default_flow_style=True)

points_dict = {}
for point in points:
    if point.valid:
        points_dict[point.track_id] = np.array([point.coord.x, point.coord.y, point.coord.z],dtype='float32')
        
calib = selected_photos[0].calibration.save('/data/Lars/Project_files/calibration_export_opencv.xml', format='opencv')

tree = ET.parse('/data/Lars/Project_files/calibration_export_opencv.xml')
root = tree.getroot()

K_string = root.getchildren()[3].getchildren()[3].text
K = np.fromstring(K_string, dtype=float,sep=" ").reshape((3,3))

D_string = root.getchildren()[4].getchildren()[3].text
D = np.fromstring(D_string, dtype=float,sep=" ")

with open("/data/Lars/Project_files/metashape_export.pkl", 'wb') as output:
    pickle.dump((frames, points_dict, K, D), output)