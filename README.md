
Current stage in replication is training 
stores wav and csv data in data/raw/
it outputs the files in data/out/
takes in raw wave files, trains and outputs best weights and performance and data evaluation(labeling). 
run entire project with:  python run.py data features model evaluate  :
deletes data directory and recreates each time the above command is ran.