# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_mean_DIM = 20
MFCC_std_DIM  = 20

def mean_std_mfcc(dataset='train'):    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(MFCC_mean_DIM + MFCC_std_DIM, 1100))
    else:
        mfcc_mat = np.zeros(shape=(MFCC_mean_DIM + MFCC_std_DIM, 300))

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # mean pooling, std pooling
        mfcc_mat[:,i]= np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)), axis=0)
        i = i + 1

    f.close()

    return mfcc_mat
        
if __name__ == '__main__':
    train_data = mean_std_mfcc('train')
    valid_data = mean_std_mfcc('valid')

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.show()








