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
import yaml
import glob
import re
import argparse
np.set_printoptions(precision=3,suppress=True)
from scipy.io import loadmat
import g2o

import numpy
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se2 = g2o.VertexSE2()
        v_se2.set_id(id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(fixed)
        super().add_vertex(v_se2)

    def add_edge(self, vertices, measurement, 
            information=np.identity(3),
            robust_kernel=None):

        edge = g2o.EdgeSE2()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()

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
    
mat_folder = 'data/tape_mats'

mats = sorted([f for f in glob.glob(path+mat_folder+'/match*') 
                 if re.match('^.*\.'+'mat'+'$', f, flags=re.IGNORECASE)])
mat_names = [m.split('/match')[-1].split('.mat')[0] for m in mats]

all_pos = []    
for m in mats:
    pos = m.split('/match')[-1].split('.mat')[0].split('_')
    if not pos[0] in all_pos:
        all_pos.append(pos[0])
    if not pos[1] in all_pos:
        all_pos.append(pos[1])

all_pos = sorted(all_pos)
links = np.zeros((len(all_pos),len(all_pos)),dtype=bool)
for i,pos1 in enumerate(all_pos):
    for j,pos2 in enumerate(all_pos):
        file_string = pos1+'_'+pos2
        if (file_string in mat_names): links[i,j]=True
H = [] 
Tw0 = np.eye(3)
Twi = Tw0

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
#fig1.subplots_adjust(0,0,1,1)
plt.get_current_fig_manager().window.setGeometry(640+window_xadj,window_yadj,640,676) #(864, 430, 800, 900)
#move_figure(position="right")
fig1.suptitle('Camera poses')
#plot_pose2_on_axes(ax1,Twi, axis_length=10.0)
H.append(Twi)

g2o_obj = PoseGraphOptimization()
g2o_obj.add_vertex(int(all_pos[0]),g2o.SE2(np.array([0,0,0])), fixed=True)

for pos1,pos2 in zip(all_pos[:-1],all_pos[1:]):
    mat_file = loadmat(mats[mat_names.index(pos1+"_"+pos2)])
    lt = mat_file['link_type'][0,0]
    ff = mat_file['ff']
    gg = mat_file['gg']
    
    _, mask	=	cv2.findHomography(	ff, gg,  cv2.RANSAC)
    rigid = cv2.estimateRigidTransform(ff,gg,False)
    R=rigid[:2,:2]
    U,S,V = np.linalg.svd(R)
    R_corr = U @ V
    rigid[:2,:2]=R_corr	
    Tji = np.vstack((rigid , np.array([0,0,1])))   
    #retval1 = np.insert(retval,2,np.zeros(3),axis=0)
    #Tji = np.insert(retval1,2,np.array([0,0,1,0]),axis=1)
    Tij = np.linalg.inv(Tji)
    thetam = np.arctan2(Tij[0,1], Tij[0,0])
    transm = Tij[:2,-1]
    
    Twj = Twi @ Tij
    Twi = Twj
    thetaw = np.arctan2(Twi[0,1], Twi[0,0])
    transw = Twi[:2,-1]
    
    #plot_pose2_on_axes(ax1,Twi, axis_length=10.0)    
    H.append(Twi)
    print(Tji, transw, thetaw)
    
    g2o_obj.add_vertex(int(pos2),g2o.SE2(np.array([*transw,thetaw])))
    g2o_obj.add_edge([int(pos1), int(pos2)],g2o.SE2(np.array([*transm,thetam])))

# Close the loop
pos1 = all_pos[0]; pos2 = all_pos[-1]
mat_file = loadmat(mats[mat_names.index(pos1+"_"+pos2)])
lt = mat_file['link_type'][0,0]
ff = mat_file['ff']
gg = mat_file['gg']

_, mask	=	cv2.findHomography(	ff, gg,  cv2.RANSAC)
rigid = cv2.estimateRigidTransform(ff,gg,False)
R=rigid[:2,:2]
U,S,V = np.linalg.svd(R)
R_corr = U @ V
rigid[:2,:2]=R_corr	
Tji = np.vstack((rigid , np.array([0,0,1])))   
Tij = np.linalg.inv(Tji)
thetam = np.arctan2(Tij[0,1], Tij[0,0])
transm = Tij[:2,-1]

Twj = Twi @ Tij
Twi = Twj
thetaw = np.arctan2(Twi[0,1], Twi[0,0])
transw = Twi[:2,-1]

#plot_pose2_on_axes(ax1,Twi, axis_length=10.0)    
H.append(Twi)
print(Tji, transw, thetaw)

#g2o_obj.add_vertex(int(pos2),g2o.SE2(np.array([*transw,thetaw])))
g2o_obj.add_edge([int(pos1), int(pos2)],g2o.SE2(np.array([*transm,thetam])))

plot_g2o_SE2(ax1,g2o_obj, text=True)
    
g2o_obj.optimize()

plot_g2o_SE2(ax1,g2o_obj, text=False)


ax1.set_aspect('equal')         # important!
