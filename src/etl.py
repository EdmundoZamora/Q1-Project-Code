import os
import urllib.request
import tarfile
from zipfile import ZipFile
import pandas as pd

def get_data(outdir):
    '''
    download and extract wav and csv data from NIPS4B.
    Handle duplicate downloading. 
    '''
    #print(outdir)
    #print(type(outdir))
    #print(os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"))

    wav_dl = "http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz"
    csv_dl = "https://figshare.com/ndownloader/files/16334603"

    tar_path, responseHeaders = urllib.request.urlretrieve(wav_dl, os.path.join(outdir, 'NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz'))
    csv_path, responseHeaders = urllib.request.urlretrieve(csv_dl, os.path.join(outdir, 'temporal_annotations_Nips4b.zip'))
    
    print(tar_path, responseHeaders)
    print(csv_path, responseHeaders)
    
    # extract tar 
    tp = tar_path
    print("Extracting wavs from tar")
    with tarfile.open(tp,'r:gz') as tf: # this part takes a while
        tf.extractall(outdir) # what is the path to this
    print("wavs from tar extracted")

    # extract zip
    cp = csv_path
    print("Extracting csv's from zip")
    with ZipFile(cp) as zf: # this needs to get moved into the NIPS folder, or take out train and test folders
        zf.extractall(os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"))
    print("csv's from zip extracted")

    # remove compressed files
    print("removing zip")
    os.remove(cp)
    print("removing tar")
    os.remove(tp)
    
    return os.path.join(outdir,"NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV")

#get_data("C:\\Users\lianl\Repositories\Methodology5\data\\raw")