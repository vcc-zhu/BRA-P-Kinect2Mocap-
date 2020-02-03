# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:13:27 2018

@author: Administrator
"""

import numpy as np
import os

import time

start = time.time()
batch_size=1
#segment_path = "./data/noise"               #原始数据文件夹

segment_path = './kinect_120/'
#segment_path = 'H:/23号待测试网络/fig8/basketball/EBD_EBF/output/'
#segment_path = "E:/data"
if not os.path.exists(segment_path):  # 要是不存在上述文件，则创建一个文件
    os.makedirs(segment_path)
files = os.listdir(segment_path)  # 得到文件夹下的所有文件名称即目录
data_file_number = len(files)  # 目录中文件的个数（数据文件的个数）
print('data_file_number',data_file_number)
segment_table_w=open('./segment_table.txt','w')#保存
for i in range(len(files)):
    segment_table_w.write(files[i]+'\n')

def get_next_batch(batch_size, each, files):  # 这里的each是一个迭代器，因为有5000个图片每次100个，所以循环传入是0到49
    batch_x = np.zeros([batch_size,120,75])
    
    def get_txt(i, each):
        
        txt_num = each * batch_size + i
        txt_name = files[txt_num]
        txt_content = np.loadtxt(segment_path + "/" + txt_name)
#        print('1',txt_content.shape)
        return txt_content
        

    for i in range(batch_size):  # 按照batch_size 迭代
        txt_content = get_txt(i, each)
        batch_x[i, :] = txt_content
    #----对数据预处理
    
    return batch_x
#a[0:2,1]表示第0行，第1行
b=np.zeros(shape=[data_file_number,120,75])
for each in range(data_file_number):
    print('each',each)
    batch_x = get_next_batch(batch_size, each, files)#batch_x.shape (1, 240, 73)
    
    b[each]=batch_x[0]
    
#    print('batch_x.shape',batch_x.shape)
#    print(batch_x )
np.savez("./rotate_kinect_120_train_August.npz", X=b)
#data = np.load('E:/tensorflow_code/spare_data/five/result.npz')['clips']加载npz文件
end = time.time()
print("程序运行时间是（秒）", (end - start))   