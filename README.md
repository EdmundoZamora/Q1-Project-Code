### Produces Binary classifications of bird vocalization presence in a wav files

### Downloads and Stores traing wav files in data/raw/ and outputs csv classifications and graph results in data/out/

### Takes in raw wave files, trains and outputs best weights and performance and data evaluation(labeling). 

### Run entire project with:  python run.py data features model evaluate  : deletes data directory and recreates each time the above command is ran.

### If data is already downloaded, spare your self the wait using : python run.py data skip features model evaluate : including skip in the targets skips the data downloading step.

### Thank you to the Engineers4Exploration organization; Ryan Kastner, Jacob Ayers and member Mugen Blue pipeline developer for and guiding us in this project. TweetyNet authors Yarden cohen,  David Nicholson, Timothy J. Gardner. NIPS4b data publisher Veronica Morfi.