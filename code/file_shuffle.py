# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:58:38 2018

@author: Administrator
"""

import numpy as np
data = np.load('./rotate_mocap_120_train_August.npz')
X_origi = data['X']

X_noise = np.load('./rotate_kinect_120_train_August.npz')['X']
rng = np.random.RandomState(23500)
I = np.arange(len(X_origi))
print(len(X_origi))
rng.shuffle(I)
X_origi = X_origi[I] #随机排序
X_noise = X_noise[I]
np.savez("./rotate_dongbu_120_train_August_shuffle.npz", X=X_origi)
np.savez("./rotate_kinect_120_train_August_shuffle.npz", X=X_noise)
#print(X.shape)