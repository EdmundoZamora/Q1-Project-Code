import os
from re import I
import sys
import csv
import math
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from pkg_resources import split_sections
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
# from torchsummary import summary

from src.TweetyNetAudio import wav2spc, create_spec, load_wav
import random
import librosa



def get_frames(x, frame_size, hop_length):
    return ((x) / hop_length) + 1 #(x - frame_size)/hop_length + 1

def frames2seconds(x, sr):
    return x/sr

def compute_windows(spc,Y,win_size):
    spc = spc
    Y=Y
    win_size = win_size
    return

def window_data(spcs, ys, uids, time_bins, windowsize):
    windowed_dataset = {"uids": [], "X": [], "Y": []}
    print("Windowing Spectrogram")
    for i in range(len(uids)):
        if ys[i].shape[0] >= windowsize//time_bins[i]:
            spc_split, Y_split, uid_split = window_spectrograms(spcs[i],ys[i], uids[i], time_bins[i], windowsize)
            windowed_dataset["X"].extend(spc_split)
            windowed_dataset["Y"].extend(Y_split)
            windowed_dataset["uids"].extend(uid_split)
    return windowed_dataset

def window_spectrograms(spc, Y, uid, time_bin, windowsize):
    computed = windowsize//time_bin #verify, big assumption. are time bins consistant?
    # print(computed*(Y.shape[0]//computed))
    time_axis = int(computed*(spc.shape[1]//computed))
    freq_axis = int(spc.shape[1]//computed) # 31, 2, 19
    #print(windowsize)
    #print("time_bin:", time_bin)
    #print(computed)
    #print(computed*(Y.shape[0]//computed))
    #print(spc.shape)
    #print(len(Y))
    #print(uid)
    #print(uid)
    #print(spc.shape, Y.shape)
    #print(time_axis, freq_axis)
    spc_split = np.split(spc[:,:time_axis],freq_axis,axis = 1)
    Y_split = np.split(Y[:time_axis],freq_axis)
    uid_split = [str(i) + "_" + uid for i in range(freq_axis)]
    return spc_split, Y_split, uid_split

def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def load_splits(spcs, ys, uids, time_bins, data_path, folder, set_type, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_{}_bin_mel_{}.pkl".format(folder, set_type))
    print(f"loading dataset for {set_type}")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else: # need to go through each element
        dataset = {"X": spcs, "Y": ys, "uids": uids, "time_bins": time_bins}
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    X = dataset["X"]#.astype(np.float32)/255
    Y = dataset["Y"]#.astype(np.longlong)
    uid = dataset["uids"]
    time_bin = dataset["time_bins"]
    print(X.shape, Y.shape, uid.shape, time_bin.shape)
    return X, Y, uid, time_bin

def new_calc_Y(sr, spc, annotation, frame_size, hop_length):
    y = [0] * spc.shape[1] # array of zeros
    for i in range(len(annotation)):
        start = get_frames(annotation.loc[i, "OFFSET"] * sr, frame_size, hop_length)
        end = get_frames((annotation.loc[i, "OFFSET"] + annotation.loc[i, "DURATION"]) * sr, frame_size, hop_length)
        for j in range(math.floor(start), math.floor(end)): #CORRECT WAY TO ADD TRUE LABELS?
            y[j] = 1 
    return y

def new_compute_Y(wav, f, spc, df, SR, frame_size, hop_length):
    #df = new_find_tags(data_path, SR, csv_file)
    wav_notes = df[df['IN FILE'] == f ]
    if os.path.isfile(wav):
        #_, sr = librosa.load(wav, sr=SR)
        annotation = wav_notes[['OFFSET','DURATION','MANUAL ID']].reset_index(drop = True)
        y = new_calc_Y(SR, spc, annotation, frame_size, hop_length)
        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]

#tags must contain ['OFFSET','DURATION','MANUAL ID'] as headers. may want to abstract this? will take a look at dataset to make this work.
def new_find_tags(csv_path, SR):
    df = pd.read_csv(csv_path, index_col=False, usecols=["IN FILE", "OFFSET", "DURATION", "MANUAL ID","SAMPLE RATE"])
    df = df[df["SAMPLE RATE"] == SR]
    return df

def new_compute_feature(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for dataset {os.path.basename(data_path)}")  
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    df = new_find_tags(csv_path, SR) # means we will have data in kaleidoscope format that would work across the whole dataset.
    valid_filenames = set(df["IN FILE"].drop_duplicates().values.tolist())
    file_path = os.path.join(data_path, folder)
    filenames = set(os.listdir(file_path))
    true_wavs = filenames.intersection(valid_filenames)
    for f in true_wavs:
        wav = os.path.join(file_path, f)
        spc,len_audio = wav2spc(wav, fs=SR, n_mels=n_mels)
        time_bins = len_audio/spc.shape[1]
        Y = new_compute_Y(wav,f, spc, df, SR, frame_size, hop_length)
        features["uids"].append(f)
        features["X"].append(spc)
        features["Y"].append(Y)
        features["time_bins"].append(time_bins)
    return features

def new_load_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_bin_mel_dataset.pkl")
    print(mel_dump_file)
    print(os.path.exists(mel_dump_file))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = new_compute_feature(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']
    return X, Y, uids, time_bins

def new_load_and_window_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, windowsize):
    x, y, uids, time_bins = new_load_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, use_dump=True)
    dataset = window_data(x, y, uids, time_bins, windowsize)

    X = np.array(dataset["X"]).astype(np.float32)/255
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    Y = np.array(dataset["Y"]).astype(np.longlong)
    # Y = Y.reshape(Y.shape[1], Y.shape[2])
    UIDS = np.array(dataset["uids"])
    return X, Y, UIDS


#Multiclass

def multiclass_calc_Y(sr, spc, annotation, manual_id_map, frame_size, hop_length):
    y = [0] * spc.shape[1] # array of zeros
    for i in range(len(annotation)):
        start = get_frames(annotation.loc[i, "OFFSET"] * sr, frame_size, hop_length)
        end = get_frames((annotation.loc[i, "OFFSET"] + annotation.loc[i, "DURATION"]) * sr, frame_size, hop_length)
        for j in range(math.floor(start), math.floor(end)): #CORRECT WAY TO ADD TRUE LABELS?
            y[j] = manual_id_map[annotation.loc[i, "MANUAL ID"]]
    return y

def multiclass_compute_Y(wav, f, spc, df, manual_id_map, SR, frame_size, hop_length):
    #df = new_find_tags(data_path, SR, csv_file)
    wav_notes = df[df['IN FILE'] == f ]
    if os.path.isfile(wav):
        #_, sr = librosa.load(wav, sr=SR)
        annotation = wav_notes[['OFFSET','DURATION','MANUAL ID']].reset_index(drop = True)
        y = multiclass_calc_Y(SR, spc, annotation, manual_id_map, frame_size, hop_length)
        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]

def multiclass_map(df):
    manual_map = {"NonBird": 0}
    manual_ids = df['MANUAL ID'].unique()
    manual_ids.sort()
    i = 1
    for id in manual_ids:
        manual_map[id]  = i
        i+=1
    return manual_map

def multiclass_compute_feature(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for dataset {os.path.basename(data_path)}")  
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    df = new_find_tags(csv_path, SR) # means we will have data in kaleidoscope format that would work across the whole dataset.
    #for multiclass: we need to map each unique manual label to a number (1-??)
    #up to the user to filter out the csv. 
    manual_id_map = multiclass_map(df)
    valid_filenames = set(df["IN FILE"].drop_duplicates().values.tolist())
    file_path = os.path.join(data_path, folder)
    filenames = set(os.listdir(file_path))
    true_wavs = filenames.intersection(valid_filenames)
    for f in true_wavs:
        wav = os.path.join(file_path, f)
        print(wav)
        spc,len_audio = wav2spc(wav, fs=SR, n_mels=n_mels)
        time_bins = len_audio/spc.shape[1]
        Y = multiclass_compute_Y(wav,f, spc, df, manual_id_map, SR, frame_size, hop_length)
        features["uids"].append(f)
        features["X"].append(spc)
        features["Y"].append(Y)
        features["time_bins"].append(time_bins)
    return features, manual_id_map

def multiclass_load_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, use_dump=True):
    mel_dump_file = os.path.join(data_path, "multi_downsampled_bin_mel_dataset.pkl")
    print(mel_dump_file)
    print(os.path.exists(mel_dump_file))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
            df = new_find_tags(csv_path, SR)
            manual_id_map = multiclass_map(df)
    else:
        dataset, manual_id_map = multiclass_compute_feature(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']
    return X, Y, uids, time_bins, manual_id_map

def multiclass_load_and_window_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, windowsize):
    x, y, uids, time_bins, manual_id_map = multiclass_load_dataset(data_path, folder, csv_path, SR, n_mels, frame_size, hop_length, use_dump=True)
    print(len(x), len(y), len(uids), len(time_bins))
    dataset = window_data(x, y, uids, time_bins, windowsize)
    X = np.array([dataset["X"]])#.astype(np.float32)/255
    X = X.reshape(X.shape[1], 1, X.shape[2], X.shape[3])
    Y = np.array([dataset["Y"]])#.astype(np.longlong)
    Y = Y.reshape(Y.shape[1], Y.shape[2])
    uid = np.array([dataset["uids"]])
    uid = uid.reshape(uid.shape[1])
    print(X.shape, Y.shape, uid.shape)
    return X, Y, uid, manual_id_map













#for a single wav file. 

def create_spec(data_path, csv_path, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for {os.path.basename(data_path)}")  
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    wav = os.path.join(data_path)
    spc,len_audio = wav2spc(wav, fs=SR, n_mels=n_mels, downsample=True)
    time_bins = len_audio/spc.shape[1]
    df = new_find_tags(csv_path, SR)
    f = os.path.basename(data_path)
    print(f)
    Y = new_compute_Y(wav, f, spc, df, SR, frame_size, hop_length)
    features["uids"].append(data_path)
    features["X"].append(spc)
    features["Y"].append(Y)
    features["time_bins"].append(time_bins)
    return features

def new_load_file(data_path, csv_path, SR, n_mels, frame_size, hop_length):
    dataset = create_spec(data_path, csv_path, SR, n_mels, frame_size, hop_length)
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']
    return X, Y, uids, time_bins


   
def load_wav_and_annotations(data_path, csv_path, SR=44100, n_mels=86, frame_size=2048, hop_length=1024, windowsize=1):
    x, y, uids, time_bins = new_load_file(data_path, csv_path, SR, n_mels, frame_size, hop_length)
    dataset = window_data(x, y, uids, time_bins, windowsize)
    X = np.array(dataset['X'])
    # print(X.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    # print(X.shape)

    uid = uid.reshape(uid.shape[1])
    Y = np.array([dataset["Y"]]).astype(np.longlong)
    Y = Y.reshape(Y.shape[1], Y.shape[2])
    UIDS = np.array([dataset["uids"]])
    UIDS = UIDS.reshape(UIDS.shape[1])
    return X, Y, UIDS


## End portion
##############

#multi class
#strongly labeled binary
#weakly labeled binary
#encorporate keeping split_sections
#we will only read kaliedoscope format
