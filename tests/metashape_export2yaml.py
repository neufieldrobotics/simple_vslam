#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract data from processed Metashape project to yaml file.
This scipts needs to be run in the Metashape Console

@author: vik748
"""

import Metashape
import numpy as np
import yaml
import pickle

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

Metashape.app.update()

print("Highlighted points for the following cameras: ")
print(selected_photos)

frames = []
for cam in selected_photos:
    kp_list=[]
    track_id_list = []
    for pj in projections[cam]:
        kp_list.append([pj.coord.x, pj.coord.y])
        track_id_list.append([pj.track_id])
    frame_data = {}
    frame_data['name'] = cam.label
    frame_data['kp'] = kp_list
    frame_data['track_ids'] = track_id_list
    
    frames.append(frame_data)
    
#with open('data.yml', 'w') as outfile:
#    yaml.dump(frames, outfile, default_flow_style=True)
with open("data.pkl", 'wb') as output:
    pickle.dump(frames, output)   