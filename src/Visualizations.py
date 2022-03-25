import librosa
from matplotlib import pyplot as plt

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
def pyha_visualization():
    return