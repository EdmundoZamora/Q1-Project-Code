import pandas as pd
import os
from tabulate import tabulate
evals = pd.read_csv(os.path.join("data/out","Evaluation_on_data.csv"))
#print(tabulate(evals, headers='keys', tablefmt='psql'))
print(evals.shape)

time = pd.read_csv(os.path.join("data/out","Time_intervals.csv"))
#print(tabulate(time, headers='keys', tablefmt='psql'))
print(time.shape)

print(f"91800/216 = {float(91800/216)}")

nu_time = pd.concat([time]*425, ignore_index=True)
#print(tabulate(nu_time, headers='keys', tablefmt='psql'))
print(nu_time.shape)

extracted_col = nu_time["start_time"]
evals_timed = evals.join(extracted_col)
evals_timed.to_csv(os.path.join("data/out","Time_interval_Evaluations.csv"))
#print(tabulate(evals_timed, headers='keys', tablefmt='psql'))
#print(evals_timed.shape)
# add three new columns, two for start and end time intervals, and one for trainfile id

# use trainfile id to pull annotation csv
# compare annotation time intervals with the binary classifications. 
# dont filter just overlay and compare
# analyse the difference, are the evaluations, spot on? within true? encapsulate true?

# using the real nips4b annotations to check if tweetynet detected in that time interval

# iteratively filter for only the trainfile in interest. 
# compare and score, add to a score chart (list)

# next trainfile

real = pd.read_csv(os.path.join("data/raw/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/temporal_annotations_nips4b","annotation_train496.csv"))
print(tabulate(real, headers='keys', tablefmt='psql'))