import os
import sys
import csv
import math
import pickle
from collections import Counter
from datetime import datetime
from graphs import file_graph_temporal, file_graph_temporal_rates
from scoring import file_score
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
# from torchsummary import summary

#region
from torch import nn
from torch.utils.data import DataLoader
from network import TweetyNet
import librosa
from librosa import display
import scipy.signal as scipy_signal
from torch.utils.data import Dataset
from glob import glob
#endregion

from TweetyNetAudio import wav2spc, create_spec, load_wav
import random
from CustomAudioDataset import CustomAudioDataset
from TweetyNetModel import TweetyNetModel

import matplotlib.pyplot as plt
from Load_data_functions import load_dataset, load_pyrenote_dataset


'''
# datasets_dir, path to training data
# folder, the title of the train file "train", in configs feature.
# what about nonBird_labels? 
# "nonBird_labels":["Plasab_song", "Unknown", "Tibtom_song", "Lyrple_song", "Plaaff_song", "Pelgra_call", "Cicatr_song", "Cicorn_song", "Tetpyg_song", "Ptehey_song"],

# what is found? 
# "found": {"Plasab_song": 0, "Unknown": 0, "Tibtom_song": 0, "Lyrple_song": 0, "Plaaff_song": 0, "Pelgra_call": 0, "Cicatr_song": 0, "Cicorn_song": 0, "Tetpyg_song": 0, "Ptehey_song": 0}
'''

#WIP
def apply_features(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found):
    train = True
    fineTuning = False
    print("----------------------------------------------------------------------------------------------")
    print("\n")
    print("IGNORE MISSING WAV FILES - THEY DONT EXIST")
    # load_data_set returns variables which get fed into model builder 

    '''
    #need
    folder = 'train'
    X, Y, uids = load_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, use_dump=True)
    print(f'X shape {X.shape}') #number of birds, rows of each data column of each data.
    print(f'len of X {len(X)}')
    bird1 = X[0] #data point, [0][0] feature value of dp, yes
    bird1 = bird1.reshape(bird1.shape[1], bird1.shape[2])
    print(bird1)
    print(f'len of bird1 {len(bird1)}') # 216
    print(f'shape of bird1 {bird1.shape}') # 216,72, r,c frequency bins, time bins
    print(f'arrays inside bird1 {len(X[0][0])}') # 72
    print(f'shape of bird1 first array {X[0][0].shape}')
    print(f'bird1 uid {uids[0]}')
    print(f'number of different birds {len(uids)}')
    spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='mel', x_axis='time') # displays rotated
    print(spec)
    plt.show()
    X_train, X_val, Y_train, Y_val, uids_train, uids_val = train_test_split(X, Y, uids, test_size=.2)
    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
    #
    # Actually subset for a test set
    #
    test_dataset = CustomAudioDataset(X, Y, uids)
    all_tags = [0,1]
    return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR
    '''

    '''
    Input~ (batch_size, num_features)
    For you num_features=len(X[0])
    Then take X[0] up to X[4] and convert them to torch tensors or bumpy arrays
    Then concatenate them into 1 tensor of size (5, feature_dimension)
    Input to the model ~ (batch_size, X[0].shape()[0], X[0].shape()[1])


    [0.24010977 0.23532039 0.19375311 0.20493513 0.20764025 0.21336952
     0.21472006 0.18257035 0.1642117  0.18358105 0.16941206 0.18236153
     0.19582106 0.17015569 0.15684327 0.19284132 0.19654424 0.2038911
     0.19246964 0.17791316 0.20639075 0.18242668 0.2063193  0.21301419
     0.1842515  0.18990998 0.1990673  0.20012796 0.19997334 0.17124568
     0.19873606 0.19232446 0.19961198 0.19416666 0.17691506 0.17580959
     0.18959488 0.1889181  0.19824806 0.19839855 0.19033328 0.1896998
     0.19550033 0.18701632 0.18685915 0.1859789  0.1825153  0.19059047
     0.18927313 0.18963294 0.18442841 0.18571979 0.19213313 0.1927422
     0.19115466 0.17871146 0.18025552 0.1843735  0.18398777 0.1811356
     0.18121892 0.18593583 0.18929477 0.18724643 0.18790492 0.18681087
     0.18502367 0.17353761 0.17283072 0.16291472 0.1656819  0.15812412]
    '''
    
    print("\n")
    print("----------------------------------------------------------------------------------------------")

    '''
    # testing area
    # find_tags(datasets_dir, folder)

    # find_pyrenote_tags(datasets_dir, folder)

    # create_pyrenote_tags(datasets_dir, folder)

    # feats = compute_pyrenote_feature(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH,2)
    # print(len(feats['uids'])) # issue
    # print(feats['uids'][0]) # string
    # print(len(feats['X'])) # list of matrices
    # print(feats['X'][0].shape)
    # print(len(feats['Y'])) # list containing same number of time bins.
    # print(len(feats['Y'][0]))
    # print(feats['Y'][0])

    # print(feats['Y'][0].shape)
    # # lengths should all be 5609
    # spec = librosa.display.specshow(feats['X'][0], hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel') # displays rotated
    # print(spec)
    # plt.show()
    # return
    ''' 

    X, Y, uids = load_pyrenote_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, 2)
    
    '''
    # print(f'X shape {X.shape}') #number of birds, rows of each data column of each data.
    # print(f'len of X {len(X)}')
    # bird1 = X[0] #data point, [0][0] feature value of dp, yes
    # print(bird1)
    # print(f'len of bird1 {len(bird1)}') # 216
    # print(f'shape of bird1 {bird1.shape}') # 216,72, r,c frequency bins, time bins
    # print(f'arrays inside bird1 {len(X[0][0])}') # 72
    # print(f'shape of bird1 first array {X[0][0].shape}')
    # print(f'bird1 uid {uids[0]}')
    # print(f'number of different birds {len(uids)}')
    # spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel') # displays rotated
    # print(spec)
    # plt.show()
    # return
    '''
    
    '''
    X shape (316,)
    len of X 316
    [[0.1471715  0.1471715  0.1471715  ... 0.1471715  0.1471715  0.1471715 ]
    [0.15273927 0.20773703 0.22108233 ... 0.23918882 0.24105152 0.25573134]
    [0.15670441 0.22601752 0.24486697 ... 0.2376636  0.25984496 0.29843926]
    ...
    [0.16158648 0.21740994 0.23785846 ... 0.24069901 0.22907388 0.24030001]
    [0.15966572 0.18686518 0.1907927  ... 0.19339544 0.15646473 0.16151345]
    [0.1471715  0.1471715  0.1471715  ... 0.1471715  0.1471715  0.1471715 ]]
    len of bird1 2666
    shape of bird1 (2666, 72)
    arrays inside bird1 72
    shape of bird1 first array (72,)
    bird1 uid Accipiter-superciliosus-329800.wav
    number of different birds 316
    '''
    
    '''
    [[0.1471715  0.1471715  0.1471715  ... 0.1471715  0.1471715  0.1471715 ]
    [0.15273927 0.20773703 0.22108233 ... 0.23918882 0.24105152 0.25573134]
    [0.15670441 0.22601752 0.24486697 ... 0.2376636  0.25984496 0.29843926]
    ...
    [0.16158648 0.21740994 0.23785846 ... 0.24069901 0.22907388 0.24030001]
    [0.15966572 0.18686518 0.1907927  ... 0.19339544 0.15646473 0.16151345]
    [0.1471715  0.1471715  0.1471715  ... 0.1471715  0.1471715  0.1471715 ]]
    [0 0 0 ... 0 0 0]
    Accipiter-superciliosus-329800.wav
    (316,)
    [[0.1471715  0.1471715  0.1471715  ... 0.1471715  0.1471715  0.1471715 ]
    [0.15273927 0.20773703 0.22108233 ... 0.23918882 0.24105152 0.25573134]
    [0.15670441 0.22601752 0.24486697 ... 0.2376636  0.25984496 0.29843926]
    ...
    [0.16158648 0.21740994 0.23785846 ... 0.24069901 0.22907388 0.24030001]
    [0.15966572 0.18686518 0.1907927  ... 0.19339544 0.15646473 0.16151345]
    [0.1471715  0.1471715  0.1471715  ... 0.1471715  0.1471715  0.1471715 ]]
    2666
    (2666, 72)
    arrays inside bird1
    72
    (72,)
    316
    Accipiter-superciliosus-329800.wav
    316
    '''
    
    '''
    # return
    # Invalid shape for monophonic audio: ndim=2, shape=(762624, 2)
    # folder = 'train'
    # compute_feature(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found)
    # return
    '''

    # need
    # test_dataset = CustomAudioDataset(X, Y, uids) #returns entire data, not sure if best to use all as testing

    #region
    # print(test_dataset.__getitem__(0)) #USEFUL
    # return 

    # pos, total = 0,0
    #remove green and red labels
    #for k in found:
        #print(k, found[k])
    
    # need
    # X, Y, uids =  random_split_to_fifty(X, Y, uids) # worth developing further.
    #endregion
    '''
    # print(X[0])
    # print(Y[0])
    # print(uids)
    # bird1 = X[0]
    # print(X[0])
    # print(len(X))
    # print(uids[0])
    # print(len(uids))
    # spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='mel', x_axis='time') # displays rotated here as well
    # print(spec)
    # plt.show()
    # return

    # need
    # for y in Y:
    #     pos += sum(y)
    #     total += len(y)

    # print(pos, total, pos/total, len(Y))

    #features above feed into below

    # all_tags = create_tags(datasets_dir, folder)
    '''

    # need
    all_tags = [0,1]

    '''
    #print(len(Counter(all_tags)))
    #for c in range(10):
    #    print(Y[c])
    #return
    '''
    
    # need
    X_train, X_val, Y_train, Y_val, uids_train, uids_val = train_test_split(X, Y, uids, test_size=.3) # Train 70% Val 30%

    X_val, X_test, Y_val, Y_test, uids_val, uids_test = train_test_split(X_val, Y_val, uids_val, test_size=.33)# val 20%, test 10%

    #region
    # print('\n')
    # print(X_train[0])
    # print('\n')
    # print(Y_train[0])
    # print('\n')
    # print(uids_train[0])
    # print('\n')

    # bird1 = X_train[0]
    # print(len(X_train))
    # print(len(Y_train))
    # print(len(uids_train))
    # # return

    # spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel') # displays rotated here as well 
    # # print(spec)
    # plt.show()

    # print('\n')
    # print(X_val[0])
    # print('\n')
    # print(Y_val[0])
    # print('\n')
    # print(uids_val[0])
    # print('\n')

    # bird1 = X_val[0]
    # print(len(X_val))
    # print(len(Y_val))
    # print(len(uids_val))
    # return

    # spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel') # displays rotated here as well 
    # print(spec)
    # plt.show()
    # return

    # return
    # print(X_train.shape, Y_train.shape, uids_train.shape)
    # # print(X_val.shape, Y_val.shape, uids_val.shape)
    #endregion

    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    #region
    # test_dataset = CustomAudioDataset(X_test[:6], Y_test[:6], uids_test[:6]) 
    # X, Y, uid = train_dataset.__getitem__(0)
    # print('\n')
    # print(X[0])
    # print('\n')
    # print(Y)
    # print('\n')
    # print(uid)
    # print('\n')

    # bird2 = X[0]
    # spec2 = librosa.display.specshow(bird2, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel')
    # plt.show()
    # return
    #endregion

    val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
    test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
    
    #region
    # train_dataset
    # val_dataset
    # X, Y, uid = val_dataset.__getitem__(0)
    # print('\n')
    # print(X[0])
    # print('\n')
    # print(Y)
    # print('\n')
    # print(uid)
    # print('\n')

    # bird2 = X[0]
    # spec2 = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel')
    # plt.show()
    # return
    #endregion
    

    return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR


def model_build( all_tags, n_mels, train_dataset, val_dataset, Skip, lr, batch_size, epochs, outdir):
    
    # if Skip:
    #     for f in os.listdir(outdir):
    #         os.remove(os.path.join(outdir, f))
    # else:   
    #     pass
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd() 
    os.chdir(outdir)

    #region
    #if torch.cuda.is_available(): #get this to work, does not detect gpu. works on tweety env(slow)
    #device = 'cpu' #torch.device('cuda:0')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #endregion
    
    #if torch.cuda.is_available(): #get this to work, does not detect gpu. works on tweety env(slow)
    device = torch.device('cpu') #'cuda:0'
    # Does not work unless there is a NVidia gpu available.
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
    else:
        name = "CPU"
    #region
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #endregion

    print(f"Using {name} ")# torch.cuda.get_device_name(0)

    print(f"Using {device} ")# torch.cuda.get_device_name(0)

    print(datetime.now().strftime("%d/%m/%Y %I:%M:%S"))

    #timebins from traindataset
    # replace input shape (1, n_mels, 86)

    tweetynet = TweetyNetModel(len(Counter(all_tags)), (1, n_mels, 86), 86, device, binary = False)
    
    # summary(tweetynet,(1, n_mels, 86))

    history, test_out, start_time, end_time, date_str = tweetynet.train_pipeline(train_dataset,val_dataset, None,
                                                                       lr=lr, batch_size=batch_size,epochs=epochs, save_me=True,
                                                                       fine_tuning=False, finetune_path=None, outdir=outdir)
    print("Training time:", end_time-start_time)

    os.chdir(cwd)

    with open(os.path.join(outdir,"nips_history.pkl"), 'wb') as f:   # where does this go??? it has to end up in data/out
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL) 

    return tweetynet, date_str


def evaluate(model,test_dataset, date_str, hop_length, sr, outdir,temporal_graphs): # How can we evaluauate on a specific wav file though?? and show time in the csv? and time on a spectrorgam? ¯\_(ツ)_/¯
    # consider the test dataset, done on all? or unseen?
    model_weights = os.path.join(outdir,f"model_weights-{date_str}.h5") # time sensitive file title
    tweetynet = model
    test_out, time_segs = tweetynet.test_load_step(test_dataset, hop_length, sr, model_weights=model_weights) 
    test_out.to_csv(os.path.join(outdir,"Evaluation_on_data.csv"))
    time_segs.to_csv(os.path.join(outdir,"Time_intervals.csv"))
    '''
    orig_stdout = sys.stdout
    sys.stdout = open(os.path.join('data/out','file_score_rates.txt'), 'w')
    file_score(temporal_graphs)
    sys.stdout.close()
    sys.stdout = orig_stdout
    file_graph_temporal(temporal_graphs) 
    file_graph_temporal_rates(temporal_graphs)
    '''
    return print("Finished Classifcation")