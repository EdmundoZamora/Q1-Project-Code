import pandas as pd
import numpy as np
from Load_data_functions import new_load_and_window_dataset, load_wav
from Visualizations import basic_visualization
from CustomAudioDataset import CustomAudioDataset
from TweetyNetModel import TweetyNetModel
import torch

def evaluate_a_wav(data_path, csv_path, model_weights_path):
    X, Y, uids = load_wav(data_path, csv_path, SR=44100, n_mels=86, frame_size=2048, hop_length=1024, windowsize=1)
    print(len(X))
    print(len(Y))
    print(len(uids))
    print(uids)

    test_dataset = CustomAudioDataset(X, Y, uids)
    #can adapt to make it work wwith GPU
    device = torch.device('cpu') #'cuda:0'
    # Does not work unless there is a NVidia gpu available.
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
    else:
        name = "CPU"
    print(f"Using {name} ")# torch.cuda.get_device_name(0)
    print(f"Using {device} ")# torch.cuda.get_device_name(0)
    tweetynet = TweetyNetModel(2, (1, 86, 43), 43, device, binary = False)
    predictions, local_scores = tweetynet.test_a_file(test_dataset, model_weights=model_weights_path, norm=True, batch_size=1, window_size=1)
    print(min(local_scores))
    print(max(local_scores))
    predictions.to_csv("test_predictions.csv")
    return local_scores, predictions