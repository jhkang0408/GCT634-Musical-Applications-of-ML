# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import librosa
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
data_path = './dataset/'
mfcc_path = './mfcc/'
perceptionbased_path = './perceptionbased/'

MFCC_DIM = 13
PERCEPTIONBASED_DIM = 5 

def mean_std_mfcc(dataset='train'):    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(MFCC_DIM*2, 1100))
    else:
        mfcc_mat = np.zeros(shape=(MFCC_DIM*2, 300))

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

def mean_std_deltamfcc(dataset='train'):    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        deltamfcc_mat = np.zeros(shape=(MFCC_DIM*2, 1100))
    else:
        deltamfcc_mat = np.zeros(shape=(MFCC_DIM*2, 300))

    i = 0
    for file_name in f:
        # load delta mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # mean pooling, std pooling        
        deltamfcc_mat[:,i]= np.concatenate((np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1)), axis=0)
        i = i + 1

    f.close()
    
    return deltamfcc_mat

def mean_std_delta2mfcc(dataset='train'):    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        delta2mfcc_mat = np.zeros(shape=(MFCC_DIM*2, 1100))
    else:
        delta2mfcc_mat = np.zeros(shape=(MFCC_DIM*2, 300))

    i = 0
    for file_name in f:
        # load delta mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)
        mfcc_delta = librosa.feature.delta(mfcc, order=2)
        
        # mean pooling, std pooling        
        delta2mfcc_mat[:,i]= np.concatenate((np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1)), axis=0)
        i = i + 1

    f.close()
    
    return delta2mfcc_mat

def mean_std_perceptionbased(dataset='train'):    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        perceptionbased_mat = np.zeros(shape=(PERCEPTIONBASED_DIM*2, 1100))
    else:
        perceptionbased_mat = np.zeros(shape=(PERCEPTIONBASED_DIM*2, 300))

    i = 0
    for file_name in f:
        # load perceptionbased file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        perceptionbased_file = perceptionbased_path + file_name
        perceptionbased = np.load(perceptionbased_file)

        # mean pooling, std pooling
        perceptionbased_mat[:,i]= np.concatenate((np.mean(perceptionbased, axis=1), np.std(perceptionbased, axis=1)), axis=0)
        i = i + 1

    f.close()

    return perceptionbased_mat 

def mean_std_deltaperceptionbased(dataset='train'):    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        perceptionbased_mat = np.zeros(shape=(PERCEPTIONBASED_DIM*2, 1100))
    else:
        perceptionbased_mat = np.zeros(shape=(PERCEPTIONBASED_DIM*2, 300))

    i = 0
    for file_name in f:
        # load perceptionbased file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        perceptionbased_file = perceptionbased_path + file_name
        perceptionbased = np.load(perceptionbased_file)
        perceptionbased_delta = librosa.feature.delta(perceptionbased)
        
        # mean pooling, std pooling
        perceptionbased_mat[:,i]= np.concatenate((np.mean(perceptionbased_delta, axis=1), np.std(perceptionbased_delta, axis=1)), axis=0)
        i = i + 1

    f.close()

    return perceptionbased_mat  

def euclidean_dist(x, y):
    """
    :param x: [m, d]
    :param y: [n, d]
    :return:[m, n]
    """
    m, n = x.shape[0], y.shape[0]    
    eps = 1e-6 

    xx = np.tile(np.power(x, 2).sum(axis=1), (n,1)) #[n, m]
    xx = np.transpose(xx) # [m, n]
    yy = np.tile(np.power(y, 2).sum(axis=1), (m,1)) #[m, n]
    xy = np.matmul(x, np.transpose(y)) # [m, n]
    dist = np.sqrt(xx + yy - 2*xy + eps)

    return dist 

def codebook_based_feature_summarization(dataset='train'):
    
    print('Feature Extraction...')
    f = open(data_path + dataset + '_list.txt','r')

    mfcc_list = []
    deltamfcc_list = []
    perceptionbased_list = []
    
    for file_name in f:
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        
        mfcc_file = mfcc_path + file_name
        perceptionbased_file = perceptionbased_path + file_name 
        
        mfcc = np.load(mfcc_file)
        perceptionbased = np.load(perceptionbased_file) 
        
        # mfcc
        mfcc_list.append(mfcc.T)
        
        # delta mfcc        
        deltamfcc_list.append(librosa.feature.delta(mfcc).T) 
        
        # perceptionbased
        perceptionbased_list.append(perceptionbased.T) 
        
    mfcc_mat = np.concatenate(np.array(mfcc_list, dtype=object), axis=0) 
    deltamfcc_mat = np.concatenate(np.array(deltamfcc_list, dtype=object), axis=0) 
    perceptionbased_mat = np.concatenate(np.array(perceptionbased_list, dtype=object), axis=0) 

    f.close() 

    descriptor = np.concatenate((mfcc_mat, deltamfcc_mat, perceptionbased_mat), axis=1) 
           
    if not os.path.exists('codebook.npy'):
        print('Codebook Construction...') 
        from sklearn.cluster import KMeans 
        k = 32
        codebook = (KMeans(n_clusters=k, random_state=0).fit(descriptor)).cluster_centers_
        np.save('codebook', codebook) 
    else:
        print('Codebook Load...')
        codebook = np.load('codebook.npy')
    
    print('Vector Quantization to encode histogram feature based on codebook...')
    histogram_features = [((np.apply_along_axis(lambda x: np.where(x==np.min(x), 1, 0), 1, euclidean_dist(np.concatenate((mfcc_list[i], deltamfcc_list[i], perceptionbased_list[i]), axis=1), codebook))).sum(axis=0)) for i in range(len(mfcc_list))]

    return np.array(histogram_features).T
        
if __name__ == '__main__':  

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)  
    
    train_X = np.concatenate((mean_std_perceptionbased(dataset='train'), mean_std_deltaperceptionbased(dataset='train'), mean_std_deltamfcc(dataset='train'), codebook_based_feature_summarization(dataset='train')), axis=0)
    valid_X = np.concatenate((mean_std_perceptionbased(dataset='valid'), mean_std_deltaperceptionbased(dataset='valid'), mean_std_deltamfcc(dataset='valid'), codebook_based_feature_summarization(dataset='valid')), axis=0) 
    
    train_X = train_X.T
    valid_X = valid_X.T

    # feature normalizaiton    
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
        
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)  
    
    # Feature Visualization     
    instruments = ['Bass', 'Brass', 'Flute', 'Guitar', 'Keyboard', 'Mallet', 'Organ', 'Reed', 'String', 'Vocal']

    plt.figure()
    plt.title('Bass')    
    plt.ylabel('Feature')
    plt.xlabel('Each Sample')
    plt.imshow((train_X[train_Y==1]).T, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+1.0f')
    
    for i in range(10):
        plt.figure()
        plt.title(instruments[i])    
        plt.imshow((train_X[train_Y==i+1]).T, interpolation='nearest', origin='lower', aspect='auto')
            
        # plt.imshow((valid_X[valid_Y==i+1]).T, interpolation='nearest', origin='lower', aspect='auto')
        # plt.colorbar(format='%+2.0f dB')
    
        plt.show()







