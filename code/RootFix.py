# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:58:38 2018

@author: Administrator
"""

import numpy as np
data = np.load('./rotate_mocap_120_train_August_shuffle.npz')
X_origi = data['X']
nums = X_origi.shape[0]
frame = X_origi.shape[1]

print(nums,frame)
X_origi = np.reshape(X_origi, [nums,frame,59,3])
origi_root = np.zeros([nums,frame,1,3])
for i in range(nums):
    for j in range(frame):
        origi_root[i,j] = X_origi[i,j,0]
origi_root_repeat = np.repeat(origi_root,59,axis=2)
X_origi[:,:,:,0] = X_origi[:,:,:,0] - origi_root_repeat[:,:,:,0]
X_origi[:,:,:,2] = X_origi[:,:,:,2] - origi_root_repeat[:,:,:,2]
X_origi = np.reshape(X_origi,[nums,frame,177])
#np.savetxt('sample.txt',X_origi[0]) 
X_noise = np.load('./rotate_kinect_120_train_August_shuffle.npz')['X']
X_noise = np.reshape(X_noise, [nums,frame,25,3])
noise_root = np.zeros([nums,frame,1,3])
for i in range(nums):
    for j in range(frame):
        noise_root[i,j] = X_noise[i,j,0]
noise_root_repeat = np.repeat(noise_root,25,axis=2)
X_noise[:,:,:,0] = X_noise[:,:,:,0] - noise_root_repeat[:,:,:,0]
X_noise[:,:,:,2] = X_noise[:,:,:,2] - noise_root_repeat[:,:,:,2]
X_noise = np.reshape(X_noise,[nums,frame,75])           
np.savez("./rotate_mocap_120_train_August_shuffle_RootFix.npz", X=X_origi)
np.savez("./rotate_kinect_120_train_August_shuffle_RootFix.npz", X=X_noise)
#print(X.shape)

#data = np.loadtxt('./52_050.txt')
#frame = data.shape[0]
#data = np.reshape(data,[frame,59,3])
#root = np.zeros([frame,1,3])
#
#for i in range(frame):
#    root[i] = data[i,0]
#root_result = np.repeat(root,59,axis=1)
#data[:,:,0] = data[:,:,0]-root_result[:,:,0]
#data[:,:,2] = data[:,:,2]-root_result[:,:,2]
#data = np.reshape(data,[frame,177])
#np.savetxt('sample.txt',data) 
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        