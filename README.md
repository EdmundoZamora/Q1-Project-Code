### Binary classifies bird vocalizations in a wav files \n

### Stores wav and csv data in data/raw/ \n it outputs results in data/out/

### Takes in raw wave files, trains and outputs best weights and performance and data evaluation(labeling). 

### Run entire project with:  python run.py data features model evaluate  : \n deletes data directory and recreates each time the above command is ran.

### If data is already downloaded, spare your self the wait using \n : python run.py data skip features model evaluate : \n including skip in the targets skips the data downloading step.