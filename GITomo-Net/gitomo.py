from pyexpat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gitomo_network import GITomo
import matplotlib.pyplot as plt
import os
import datetime
import copy

def my_loss_nmse(x_real, x_gt_real):

    criterion1 = nn.MSELoss()

    return criterion1(x_real, x_gt_real)

def now_to_date(format_string="%Y-%m-%d-%H-%M-%S"):
    time_stamp = int(time.time())
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


## 模型保存
def save_my_model(net, optimizer, learning_rate, dir_path = 'test/',
                  Path_Model='/data1/lch/Ada-Lista/ada-Lista-of-net/Model_save/'):
    print('===> Saving models...')
    str_date = now_to_date()
    state = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': learning_rate,
    }

    path_checkpoint = Path_Model  + dir_path
    if not os.path.isdir(path_checkpoint):
        os.makedirs(path_checkpoint)

    File_path = path_checkpoint + 'Ada-LISTA' + str_date +  '.pt'
    torch.save(state, File_path)
    print('===> Saving model finished...')
    return File_path



def GITomo_Test(model_path, useGPU, TestData,batch_size=128, max_iter =30):

    model_path = model_path
    test_D = TestData['TestData_A']
    test_Y = TestData['TestData_Y_CN']

    New_A_part1 = np.append(test_D.real,-test_D.imag,axis = 2)
    New_A_part2 = np.append(test_D.imag,test_D.real,axis = 2)
    New_A = np.append(New_A_part1,New_A_part2,axis = 1)

    test_D = New_A
    test_Y = np.append(test_Y.real,test_Y.imag,axis = 1)
    test_Y = torch.from_numpy(test_Y)

    
    #参数设置
    k, n, m = test_D.shape
    n_samples = test_Y.shape[0]
    batch_size = batch_size
    steps_per_test = n_samples // batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    test_D = torch.from_numpy(test_D)
    
    net = GITomo(n, m, max_iter=max_iter, L=1, theta=1)

    net.load_state_dict(state_dict['model_state_dict'])
    if useGPU == 1:
        net.float().cuda()

    loss_nmse_test = 0
    X_result = torch.from_numpy(np.zeros([n_samples, int(m/2)]))
    with torch.no_grad():
        for step in range(steps_per_test):

            if step%100000 == 0:
                print(step)

            Y_batch = test_Y[step*batch_size:(step+1)*batch_size]
            D_batch = test_D[step*batch_size:(step+1)*batch_size]

            if useGPU == 1:
                Y_batch = Y_batch.float().cuda()
                D_batch = D_batch.float().cuda()

            X_h = net(Y_batch,D_batch)
            X_result[step*batch_size:(step+1)*batch_size,:] = X_h

        X_result = X_result.cpu().detach().numpy()



    print('Test_result: %.5f ' % (loss_nmse_test))
    return X_result, loss_nmse_test

