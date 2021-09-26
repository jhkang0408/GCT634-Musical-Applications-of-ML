# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa

data_path = './dataset/'
mfcc_path = './mfcc/'
perceptionbased_path = './perceptionbased/'

MELBIN_DIM = 128
MFCC_DIM = 13
PERCEPTIONBASED_DIM = 5

def extract_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        y, sr = librosa.load(file_path, sr=22050)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MELBIN_DIM,fmax=8000)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=MFCC_DIM)                
        
        file_name = file_name.replace('.wav','.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc)

    f.close();

def extract_perceptionbased(dataset='train'):
    '''
        1. Zero-crossing rate
        2. Root mean square 
        3. Spectral centroid
        4. Bandwidth
        5. Flux
    '''    
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        y, sr = librosa.load(file_path, sr=22050)

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        flux = np.matrix(librosa.onset.onset_strength(y=y, sr=sr))
        
        perceptionbased = np.concatenate((zcr, rms, cent, spec_bw, flux), axis=0)        
        
        file_name = file_name.replace('.wav','.npy')
        save_file = perceptionbased_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, perceptionbased)

    f.close()
    
    return

if __name__ == '__main__':
    extract_mfcc(dataset='train')                 
    extract_mfcc(dataset='valid')            
    extract_perceptionbased(dataset='train')
    extract_perceptionbased(dataset='valid')
                    
