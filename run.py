import sys
import os 
import json 


sys.path.insert(0, 'src')

# file that gets the data DONE
# function that applies the features DONE
# function that builds the model DONE
# results go into separate files 
# perform on a docker container <-- current challenge

import env_setup
from etl import get_data
from NIPS_training import apply_features, model_build, evaluate

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'data' in targets:      
        env_setup.make_datadir() # removes each time, handles manually deleting                     
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        data = get_data(**data_cfg)

    if 'features' in targets:
        with open('config/load_dataset-params.json') as fh:
            feats_cfg = json.load(fh)

        all_tags, n_mels, train_dataset, val_dataset, test_dataset  = apply_features(data, **feats_cfg)

    if 'model' in targets:                              
        with open('config/nips-model-params.json') as fh:
            model_cfg = json.load(fh)
        # make the data target, set outputs to data/temp though, only pickle works, managed to get most, 1 missing
        model, date_str = model_build(all_tags, n_mels, train_dataset, val_dataset, **model_cfg)
    
    if 'evaluate' in targets:                              
        with open('config/evaluate-params.json') as fh:
            eval_cfg = json.load(fh)
        # evaluates and stores csvs to out/
        evaluate(model, test_dataset, date_str, **eval_cfg)

    return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
