###
#  From the Audio_Data_Augmentation.ipynb in PyHa by Jacob Ayers
###

import matplotlib.pyplot as plt
import librosa
import numpy as np
import scipy
import os
from scipy.io import wavfile
import sox
import colorednoise as cn
from multipledispatch import dispatch 

#   Create directories for the original data and the augmented data. The original data directory should have subdirectories for the different datasets. 
#   Each subdirectory contains wav files. The augmented data directory can be empty and will be populated as this notebook runs.

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        
def save(signal, sample_rate, new_dir):
    make_dir(aug_dir + dataset_name + '/' + new_dir)
    new_file_path = aug_dir + dataset_name + '/' + new_dir + '/' + filename
    if not os.path.isfile(new_file_path):
        wavfile.write(new_file_path, sample_rate, signal)
        
@dispatch(str, object, object, object) 
def augment_and_save(feature_name, augment_function, signal, sample_rate):
    if not os.path.isfile(aug_dir + dataset_name + '/' + feature_name + '/' + filename):
        save(augment_function(signal, sample_rate), sample_rate, feature_name)
    else:
        print('Augmented file already exists')
        
@dispatch(str, object, object, object, object) 
def augment_and_save(feature_name, augment_function, signal, sample_rate, factor):
    if not os.path.isfile(aug_dir + dataset_name + '/' + feature_name + '_' + str(factor) + '/' + filename):
        save(augment_function(signal, sample_rate, factor), sample_rate, feature_name + '_' + str(factor))
    else:
        print('Augmented file already exists')

# Pitch factor should be between 0.9 and 1.1
def augment_pitch(signal, sample_rate, factor):
    print("Pitch Modulation Factor: ", factor)
    pitch_modulated_signal = librosa.effects.pitch_shift(signal, sample_rate, factor)
    return pitch_modulated_signal

# Noise factor should be between 0.001 and 0.02
def augment_noise(signal, sample_rate, factor):
    print("Noise Modulation Factor: ", factor)
    noise = np.random.randn(len(signal)) 
    noise_modulated_signal = signal + factor * noise
    noise_modulated_signal = noise_modulated_signal.astype(type(signal[0]))
    return noise_modulated_signal

# Speed factor should be between 0.9 and 1.1
def augment_speed(signal, sample_rate, factor):
    print("Speed Modulation Factor: ", factor)
    speed_modulated_signal = librosa.effects.time_stretch(signal, factor)
    return speed_modulated_signal
#Will need to adjust strong labels to make sure they match up with the new speed.

# Tempo factor should be between 0.9 and 1.1
def augment_tempo_and_save(filepath, factor):
    new_file_path = aug_dir + dataset_name + '/tempo_' + str(factor) + '/' + filename
    if not os.path.isfile(new_file_path):
        print("Tempo Modulation Factor: ", factor)
        tempoTransformer = sox.Transformer()
        tempoTransformer.tempo(factor)
        new_dir = 'tempo_' + str(factor) + '/' 
        make_dir(aug_dir + dataset_name + '/' + new_dir)
        tempoTransformer.build(filepath, new_file_path)
        
# Exponent factor should be 1 for pink noise
def add_colored_noise(signal, sample_rate, factor):
    print("Gaussian distributed noise with exponent: ", factor)
    noise = cn.powerlaw_psd_gaussian(factor, sample_rate)
    noise = np.tile(noise, int(len(signal) / len(noise)) + 1)
    noise = noise[:len(signal)]
    noise_modulated_signal = signal + noise
    noise_modulated_signal = noise_modulated_signal.astype(type(signal[0]))
    return noise_modulated_signal

def add_gaussian_noise(signal, sample_rate):
    print("Gaussian noise")
    noise_modulated_signal = signal + np.random.normal(0, 0.1, signal.shape)
    noise_modulated_signal = noise_modulated_signal.astype(type(signal[0]))
    return noise_modulated_signal

def augment_data(dataset_name, filename, sr):
    filepath = orig_dir + dataset_name + '/' + filename
    signal, sample_rate = librosa.load(filepath, sr)
    
    # Add augmentations here
    augment_and_save('pitch', augment_pitch, signal, sample_rate, 1.1)
    augment_and_save('noise', augment_noise, signal, sample_rate, 0.02)
    augment_and_save('speed', augment_speed, signal, sample_rate, 1.1)
    augment_and_save('colored_noise', add_colored_noise, signal, sample_rate, 1)
    augment_and_save('gaussian_noise', add_gaussian_noise, signal, sample_rate)
    #augment_tempo_and_save(filepath, 1.1)
    
    """
    # Example of how to augment both pitch and noise
    pitch_factor = 1.1
    noise_factor = 0.02
    s = augment_pitch(signal, sample_rate, pitch_factor)
    s = augment_noise(signal, sample_rate, noise_factor)
    save(s, sample_rate, 'pitch_%s_noise_%s' % (pitch_factor, noise_factor))
    """


def main():
    orig_dir = './original_data/'
    aug_dir = './augmented_data/'
    sample_rates = {"xenocanto": 384000}

    for subdir in [x[0] for x in os.walk(orig_dir)][1:]:
        dataset_name = subdir.split('/')[-1]
        make_dir(aug_dir + dataset_name)

        for filename in os.listdir(subdir):
            if filename.endswith(".wav"):
                print(subdir + filename)
                augment_data(dataset_name, filename, sample_rates[dataset_name])
                print()
main()
