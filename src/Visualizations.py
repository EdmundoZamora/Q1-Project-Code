from dataclasses import dataclass
import librosa
from matplotlib import pyplot as plt
from pyha_visualization import local_line_graph
from TweetyNetAudio import load_wav
import pandas as pd
import numpy as np
import os
import sys

#Makes sure the spectrogram looks correct
def basic_visualization(X, SR, n_mels, frame_size, hop_length, windowsize):
    print(f'X shape {X.shape}') #number of birds, rows of each data column of each data.
    print(f'len of X {len(X)}')
    bird1 = X
    bird1 = bird1.reshape(bird1.shape[0], bird1.shape[1])
    spec = librosa.display.specshow(bird1, hop_length = hop_length,sr = SR, y_axis='mel', x_axis='time') # displays rotated
    print(spec)
    plt.show() 

#make sure annotations line up
def pyha_visualization(local_scores, wav_path, csv_path):
    SR, signal = load_wav(data_path)
    filename = os.path.basename(wav_path)
    ## do this step before to work with    
    #SR, signal, Wav_path = get_wav(filepath)
    automated_df = pd.DataFrame()
    premade_annotations_df = None # get_premade_annotations(filename, csv, dataset_type)
                            #should just load it fresh from the dataset
                            #make this only compatible with full audio.
    premade_annotations_label = "Bird"
    #print(premade_annotations_df)
    
    # change local scores to what tweetynet returns.
    log_scale = True
    save_fig = False
    normalize_local_scores = False
    local_line_graph(local_scores,
        wav_path,
        SR,
        samples=signal,
        automated_df= automated_df,
        premade_annotations_df=premade_annotations_df,
        premade_annotations_label=premade_annotations_label,
        log_scale=log_scale,
        save_fig=save_fig,
        normalize_local_scores=normalize_local_scores)



def create_visualization(filename, csv, dataset_type):
    SR, SIGNAL, Wav_path = get_wav(filename, dataset_type)
    #automated_annotations_path = os.path.join("data", "out", "Evaluation_on_data.csv")
    ####This is the part that we removed from pyha that uses microfaune.  

    #if automated_annotations_path == "":
    #    automated_df = pd.DataFrame()
    #else:
    #    automated_df = pd.read_csv(automated_annotations_path)
    #    automated_df = kaliedoscope_format(automated_df, filename)
    #print(automated_df)
    premade_annotations_df = get_premade_annotations(filename, csv, dataset_type)
    premade_annotations_label = "Bird"
    #print(premade_annotations_df)
    
    # change local scores to what tweetynet returns.
    log_scale = True
    save_fig = False
    normalize_local_scores = False
    local_line_graph([0] * 43,
        Wav_path,
        SR,
        samples=SIGNAL,
        automated_df= automated_df,
        premade_annotations_df=premade_annotations_df,
        premade_annotations_label=premade_annotations_label,
        log_scale=log_scale,
        save_fig=save_fig,
        normalize_local_scores=normalize_local_scores)

def get_wav(filename, dataset_type):
    if dataset_type =="NIPS":
        Wav_path = os.path.join("data","raw","NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV","train", filename)
        SR, SIGNAL = load_wav(Wav_path)
        return SR, SIGNAL, Wav_path
    elif dataset_type == "PYRE":
        Wav_path = os.path.join("data","PYRE","Mixed_Bird-20220126T212121Z-003", "Mixed_Bird", filename)
        SR, SIGNAL = load_wav(Wav_path)
        return SR, SIGNAL, Wav_path
    else:
        print(f"Dataset: {dataset_type} does not exist")
        return None
    

def get_premade_annotations(filename, csv, dataset_type):
    if dataset_type =="NIPS":
        premade_annotations_path = os.path.join("data","raw","NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV","temporal_annotations_nips4b", csv)
        premade_annotations_df = pd.read_csv(premade_annotations_path, names=["OFFSET", "DURATION", "TAG"])
        return premade_annotations_df
    elif dataset_type == "PYRE":
        premade_annotations_path = os.path.join("data", "PYRE", csv)
        premade_annotations_df = pd.read_csv(premade_annotations_path)
        premade_annotations_df = premade_annotations_df[premade_annotations_df["IN FILE"] == filename]
        if len(premade_annotations_df) == 0:
            print(f"No premade annotations for file: {filename}")
        premade_annotations_df = premade_annotations_df[["OFFSET", "DURATION", "MANUAL ID"]]
        return premade_annotations_df
    else:
        print(f"Dataset: {dataset_type} does not exist")
        return None

    #for kaliedoscope format
def new_get_premade_annotations(filename, premade_annotations_path):
    premade_annotations_df = pd.read_csv(premade_annotations_path)
    premade_annotations_df = premade_annotations_df[premade_annotations_df["IN FILE"] == filename]
    if len(premade_annotations_df) == 0:
        print(f"No premade annotations for file: {filename}")
    premade_annotations_df = premade_annotations_df[["OFFSET", "DURATION", "MANUAL ID"]]
    return premade_annotations_df


def kaliedoscope_format(df, filename):
    filtered_df = df[df["file"] == filename]
    if filtered_df.empty == True:
        print(f"file: {filename} was not in test set")
        return filtered_df
    sorted_filtered_df = filtered_df.sort_values("overall frame number")
    #print(sum(sorted_filtered_df["pred"]))
    time_bin_seconds = sorted_filtered_df.iloc[1]["temporal_frame_start_times"]
    #print(time_bin_seconds)
    zero_sorted_filtered_df = sorted_filtered_df[sorted_filtered_df["pred"] == 0]
    offset = zero_sorted_filtered_df["temporal_frame_start_times"]
    duration = zero_sorted_filtered_df["temporal_frame_start_times"].diff().shift(-1)
    intermediary_df = pd.DataFrame({"OFFSET": offset, "DURATION": duration})
    kaliedoscope_df = []
    if offset.iloc[0] != 0:
        kaliedoscope_df.append(pd.DataFrame({"OFFSET": [0], "DURATION": [offset.iloc[0]]}))
    kaliedoscope_df.append(intermediary_df[intermediary_df["DURATION"] >= 2*time_bin_seconds])
    if offset.iloc[-1] < sorted_filtered_df.iloc[-1]["temporal_frame_start_times"]:
        kaliedoscope_df.append(pd.DataFrame({"OFFSET": [offset.iloc[-1]], "DURATION": [sorted_filtered_df.iloc[-1]["temporal_frame_start_times"] + 
                                sorted_filtered_df.iloc[1]["temporal_frame_start_times"]]}))
    kaliedoscope_df = pd.concat(kaliedoscope_df)
    kaliedoscope_df = kaliedoscope_df.reset_index(drop=True)
    return kaliedoscope_df
    