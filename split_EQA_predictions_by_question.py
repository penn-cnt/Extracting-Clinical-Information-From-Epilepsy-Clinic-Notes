import json
import argparse
import pandas as pd
import numpy as np

#Input arguments
parser = argparse.ArgumentParser(description='Splits the prediction file from the extractive question answering model (SQuAD format) into a pred file for PQFs, and a pred file for ELOs')
parser.add_argument("--ground_truth_path", help='The path to the ground truth dataset (.json)',\
                    required=True)
parser.add_argument("--model_pred_path", help="The path to the predictions (.json)",\
                    required=True)
parser.add_argument("--output_path", help="The path to the directory to where to save the two predictions",\
                    required=True)
args = parser.parse_args()

#open the ground truth and predictions
with open(args.ground_truth_path) as f:
    dataset = json.load(f)['data']
with open(args.model_pred_path) as f:
    predictions = json.load(f)
    
#dictionaries to hold the separated predictions
PQF_predictions = {}
ELO_predictions  = {}

for datum in dataset:
    #check if this datum's id is in the predictions
    if datum['id'] not in predictions:
        continue

    #check if it is a szfreq or last occurrence question and
    #put the prediction into the right dictionary
    if datum['question'] == 'How often does the patient have events?': 
        PQF_predictions[datum['id']] = predictions[datum['id']]
    else:
        ELO_predictions[datum['id']] = predictions[datum['id']]
        
with open(args.output_path+r"/PQF_predictions.json",'w') as f:
    json.dump(PQF_predictions, f)
with open(args.output_path+r'/ELO_predictions.json', 'w') as f:
    json.dump(ELO_predictions, f)