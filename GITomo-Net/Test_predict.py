#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lch
    该文件读取数据进行网络训练

"""
from scipy.io import loadmat
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from scipy.special import comb, perm
from scipy.io import savemat
from gitomo import GITomo_Test
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


# Network para
batch_size = 128
stage = 30
numIter = 100
if torch.cuda.is_available() == True:
    useGPU = 1
import h5py

# loading
import datetime
model_path = "/data1/lch/Geometry-Independent-Network/PaperToGithub/GITomo-Net/Model_save/GITomo-Simdata-Trained-model/GITomo_Net_trained_modelpara.pt"
import hdf5storage

path_test = '/data1/lch/Geometry-Independent-Network/PaperToGithub/GITomo-Net/Matlab2Net/Data_and_para_Lbd.mat'
m = h5py.File(path_test)
g_all = m['g_re_all']
TestData_Y_CN = g_all['real'] + 1j*g_all['imag']
TestData_Y_CN = TestData_Y_CN.T

L_all = m['L_all']
TestData_A = L_all['real'] + 1j*L_all['imag']
TestData_A = TestData_A.transpose(2,1,0)

TestData_Y_CN = TestData_Y_CN

TestData_Y_CN_phase = np.exp(1j*np.angle(TestData_Y_CN))

TestData_Y_CN = TestData_Y_CN_phase

TestData_eig = np.zeros([TestData_A.shape[0],1])
for j in range(TestData_A.shape[0]):
    A = TestData_A[j,:,:]
    ATA = A@A.T.conj()
    feature_value, feature_vetor = np.linalg.eig(ATA)
    TestData_eig[j] = max(abs(feature_value))

TestData_eig_A = np.expand_dims(TestData_eig,2)


TestData_A = TestData_A/ np.sqrt(TestData_eig_A)


TestData = {'TestData_Y_CN':TestData_Y_CN, 'TestData_A':TestData_A}

X_result, loss_nmse_test = GITomo_Test(model_path, useGPU, TestData,batch_size=512, max_iter = stage)

hdf5storage.savemat('/data1/lch/Geometry-Independent-Network/PaperToGithub/GITomo-Net/Net2Matlab/GITomo-simL-result.mat' ,{'X_result':X_result},do_compression=True,format='7.3')
   
