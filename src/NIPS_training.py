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
from Load_data_functions import load_dataset, load_pyrenote_dataset, load_pyrenote_splits, load_splits


'''
# datasets_dir, path to training data
# folder, the title of the train file "train", in configs feature.
# what about nonBird_labels? 
# "nonBird_labels":["Plasab_song", "Unknown", "Tibtom_song", "Lyrple_song", "Plaaff_song", "Pelgra_call", "Cicatr_song", "Cicorn_song", "Tetpyg_song", "Ptehey_song"],

# what is found? 
# "found": {"Plasab_song": 0, "Unknown": 0, "Tibtom_song": 0, "Lyrple_song": 0, "Plaaff_song": 0, "Pelgra_call": 0, "Cicatr_song": 0, "Cicorn_song": 0, "Tetpyg_song": 0, "Ptehey_song": 0}
'''

#WIP
def apply_features(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, window_size, dataset):
    train = True
    fineTuning = False
    print("----------------------------------------------------------------------------------------------")
    print("\n")
    print("IGNORE MISSING WAV FILES - THEY DONT EXIST")
    # load_data_set returns variables which get fed into model builder 
    if dataset == "NIPS":
        folder = 'train'
        X, Y, uids = load_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, use_dump=True)
        #print(f'X shape {X.shape}') #number of birds, rows of each data column of each data.
        #print(f'len of X {len(X)}')
        #bird1 = X[0] #data point, [0][0] feature value of dp, yes
        #bird1 = bird1.reshape(bird1.shape[1], bird1.shape[2])
        #print(bird1)
        #print(f'len of bird1 {len(bird1)}') # 216
        #print(f'shape of bird1 {bird1.shape}') # 216,72, r,c frequency bins, time bins
        #print(f'arrays inside bird1 {len(X[0][0])}') # 72
        #print(f'shape of bird1 first array {X[0][0].shape}')
        #print(f'bird1 uid {uids[0]}')
        #print(f'number of different birds {len(uids)}')
        #spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='mel', x_axis='time') # displays rotated
        #print(spec)
        #plt.show()
        X_train, X_val, Y_train, Y_val, uids_train, uids_val = train_test_split(X, Y, uids, test_size=.3)
        X_val, X_test, Y_val, Y_test, uids_val, uids_test = train_test_split(X_val, Y_val, uids_val, test_size=.66)
        X_train, Y_train, uids_train = load_splits(X_train, Y_train, uids_train, datasets_dir, folder, "train")
        X_val, Y_val, uids_val = load_splits(X_val, Y_val, uids_val, datasets_dir, folder, "val")
        X_test, Y_test, uids_test = load_splits(X_test, Y_test, uids_test, datasets_dir, folder, "test")
        train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
        val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
        test_dataset = CustomAudioDataset(X, Y, uids)
        all_tags = [0,1]
        return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR
    elif dataset == "PYRE":
        X, Y, uids, time_bins = load_pyrenote_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH)
        all_tags = [0,1]
        # need
        #Split by file
        pre_X_train, pre_X_val, pre_Y_train, pre_Y_val, pre_uids_train, pre_uids_val, pre_time_bins_train, pre_time_bins_val = train_test_split(X, Y, uids, time_bins, test_size=.3) # Train 70% Val 30%

        pre_X_val, pre_X_test, pre_Y_val, pre_Y_test, pre_uids_val, pre_uids_test, pre_time_bins_val, pre_time_bins_test= train_test_split(pre_X_val, pre_Y_val, pre_uids_val, pre_time_bins_val, test_size=.66)# val 10%, test 20%

        #window spectrograms
        X_train, Y_train, uids_train, = load_pyrenote_splits(pre_X_train, pre_Y_train, pre_uids_train, pre_time_bins_train, window_size, datasets_dir, folder, "train")
        X_val, Y_val, uids_val, = load_pyrenote_splits(pre_X_val, pre_Y_val, pre_uids_val, pre_time_bins_val, window_size, datasets_dir, folder, "val")
        X_test, Y_test, uids_test, = load_pyrenote_splits(pre_X_test, pre_Y_test, pre_uids_test, pre_time_bins_test, window_size, datasets_dir, folder, "test")
        train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
        val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
        test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
        return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR
    return None


def model_build( all_tags, n_mels, train_dataset, val_dataset, Skip, time_bins, lr, batch_size, epochs, outdir, ):
    
    #if Skip:
    #    for f in os.listdir(outdir):
    #        os.remove(os.path.join(outdir, f))
    #else:   
    #    pass
    
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

    tweetynet = TweetyNetModel(len(Counter(all_tags)), (1, n_mels, time_bins), time_bins, device, binary = False)
    
    # summary(tweetynet,(1, n_mels, 86))

    history, start_time, end_time, date_str = tweetynet.train_pipeline(train_dataset,val_dataset,
                                                                       lr=lr, batch_size=batch_size,epochs=epochs, save_me=True,
                                                                       fine_tuning=False, finetune_path=None, outdir=outdir)
    print("Training time:", end_time-start_time)

    os.chdir(cwd)

    with open(os.path.join(outdir,"nips_history.pkl"), 'wb') as f:   # where does this go??? it has to end up in data/out
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL) 

    return tweetynet, date_str


def evaluate(model,test_dataset, date_str, hop_length, sr, outdir,temporal_graphs, window_size): # How can we evaluauate on a specific wav file though?? and show time in the csv? and time on a spectrorgam? ¯\_(ツ)_/¯
    # consider the test dataset, done on all? or unseen?
    model_weights = os.path.join(outdir,f"model_weights-{date_str}.h5") # time sensitive file title
    tweetynet = model
    test_out, time_segs = tweetynet.test_load_step(test_dataset, hop_length, sr, model_weights=model_weights, window_size=window_size) 
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