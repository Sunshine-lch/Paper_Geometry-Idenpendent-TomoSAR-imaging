#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lch
    该文件进行产生并初始化训练、验证、测试数据
"""
from lib2to3.pgen2.token import AMPER
from scipy.io import loadmat
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from scipy.special import comb, perm
from scipy.io import savemat

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

train_number = 2000000 # 训练数据量
valid_number = 100000 # 验证数据量
DataDir_path = '/data1/lch/Geometry-Independent-Network/PaperToGithub/GITomo-Net/Traning_data/'## data-save

if os.path.exists(DataDir_path) == 0:
    os.makedirs(DataDir_path)

def GITomo_produce_data(data_num, Target_num=4):
    
    v = np.zeros([20,1])
    s_cu = np.linspace(-5, 12, 86)

    index_row = len(v)
    index_col = len(s_cu)

    X_complex = np.zeros([data_num,index_col]) + 1j * np.zeros([data_num,index_col])
    X_complex_noise = np.zeros([data_num,index_col]) + 1j * np.zeros([data_num,index_col])
    
    Y_complex = np.zeros([data_num,index_row]) + 1j * np.zeros([data_num,index_row]) 
    Y_complex_noise = np.zeros([data_num,index_row]) + 1j * np.zeros([data_num,index_row]) 

    A_all = np.zeros([data_num,index_row,index_col]) + 1j * np.zeros([data_num,index_row,index_col])

    for i in range(data_num):
        if i%10000 == 0:
            print(i)

       
        rdlen =np.random.random(18)
        
        v = np.zeros([20,1])
        v[0] = 0
        v[-1] = 1
        v[1:-1] = np.expand_dims(rdlen,1)
        v = np.sort(v,0)

        A = np.exp(-1j*2*np.pi*v*s_cu)
                        
        sparsity = rd.sample( range( 1, Target_num) , 1 )[0]
        idx = rd.sample(list(range(index_col)), sparsity)
        X_complex[i,idx] = (1+10000*np.random.random(1)) * np.exp(1j*2*math.pi*np.random.random(sparsity))  

        SNR = 2000 * np.random.random(1) #20
        A_random = A 
        Y_complex[i,:] = np.matmul(A_random, X_complex[i, :])

        powerY = np.linalg.norm(Y_complex[i,:])**2/len(Y_complex[i,:])/2
        noise_variance = powerY/(10**(SNR/10.0))
        noise_y = np.sqrt(noise_variance/2)*np.random.randn(len(Y_complex[i,:])) + 1j * np.sqrt(noise_variance/2)*np.random.randn(len(Y_complex[i,:]))

        Y_noise = Y_complex[i,:] + noise_y
        A_all[i,:,:] = A

    return Y_complex, Y_complex_noise, X_complex, A_all


#train_data
TrianData_Y_C, TrianData_Y_CN, TrianData_X_C, TrianData_A = GITomo_produce_data(train_number,Target_num=4)

##val_data
ValidData_Y_C, ValidData_Y_CN, ValidData_X_C, ValidData_A = GITomo_produce_data(valid_number,Target_num=4)

import scipy.io as sio
import hdf5storage


hdf5storage.savemat(DataDir_path + 'train.mat', {"TrianData_Y_C":TrianData_Y_C, \
    "TrianData_Y_CN":TrianData_Y_CN, "TrianData_X_C":TrianData_X_C, "TrianData_A":TrianData_A},do_compression=True,format='7.3')
hdf5storage.savemat(DataDir_path + 'valid.mat', {"ValidData_Y_C":ValidData_Y_C,  \
    "ValidData_Y_CN":ValidData_Y_CN, "ValidData_X_C":ValidData_X_C, "ValidData_A":ValidData_A},do_compression=True,format='7.3')
