import json
import argparse
import pandas as pd
import numpy as np

#Input arguments
parser = argparse.ArgumentParser(description='Combines the prediction filse with the ground truth dataset files')
parser.add_argument("--szFreq_ground_truth_path", help='The path to the szFreq ground truth dataset (.json)',\
                    required=True)
parser.add_argument("--szFreq_predictions_path", help="The path to the szFreq predictions (.json)",\
                    required=True)
parser.add_argument("--hasSz_ground_truth_path", help="The path to the hasSz ground truth dataset (.json)",\
                    required=True)
parser.add_argument("--hasSz_predictions_path", help="The path to the hasSz predictions (.tsv)", \
                    required=True)
parser.add_argument("--output_path", help="The path to where to save the combined dataset",\
                    required=True)
args = parser.parse_args()

#open the szFreq ground truth and predictions
with open(args.szFreq_ground_truth_path) as f:
    szFreq_dataset = json.load(f)['data']
with open(args.szFreq_predictions_path) as f:
    szFreq_predictions = json.load(f)
    
#open the hasSz ground truth and predictions
with open(args.hasSz_ground_truth_path) as f:
    hasSz_dataset = []
    for line in f:
        hasSz_dataset.append(json.loads(line))
hasSz_predictions = pd.read_csv(args.hasSz_predictions_path, sep='\t', header=0)
hasSz_truth_labels = hasSz_predictions['True_Label'].tolist()
hasSz_predictions_str = ["".join(ch for ch in pred if ch not in '[]') for pred in hasSz_predictions['Predictions']]
hasSz_predictions_split = [[float(confidence) for confidence in pred.split()] for pred in hasSz_predictions_str]
hasSz_predictions_labels = [np.argmax(pred) for pred in hasSz_predictions_split]

#onehot-encoder
one_hot = {"true":1, "false":0, "no-answer":2, 'yes':1, 'no':0, '-1':2}
one_hot_reversed = {1:'true', 2:'no-answer', 0:'false'}
    
#initialize an empty list
#this list will hold a dictionary of the combined ground truths and predictions in the format:
#   {'passage':str, 'doc_name':str, 'hasSz_q':str, 'hasSz_gt':str, 'hasSz_pred':str
#    'PQF_q':str, 'PQF_gt':[str], 'PQF_pred':str, 'PQF_id':str,
#    'ELO_q':str, 'ELO_gt':[str], 'ELO_pred':str, 'ELO_id':str}   
combined_data = []

#iterate through the szFreq dataset and build up the combined list
for szFreq_datum in szFreq_dataset:
    
    has_entry = False
    #check if this passage and document already exists in combined_data. If it does,
    #add to that entry
    #otherwise, make a new entry
    for combined_datum in combined_data:
        if szFreq_datum['context'] == combined_datum['passage'] and szFreq_datum['title'] == combined_datum['doc_name']:
            
            #if it does exist, check if this is a ELO or PQF question 
            #and add the info into the entry
            has_entry = True
            if szFreq_datum['question'] == 'How often does the patient have events?':
                combined_datum['PQF_q'] = szFreq_datum['question']
                combined_datum['PQF_id'] = szFreq_datum['id'] 
                combined_datum['PQF_gt'] = []
                for answer in szFreq_datum['answers']['text']:
                    combined_datum['PQF_gt'].append(answer)
                if not combined_datum['PQF_gt']:
                    combined_datum['PQF_gt'] = ""
            else:
                combined_datum['ELO_q'] = szFreq_datum['question']
                combined_datum['ELO_id'] = szFreq_datum['id'] 
                combined_datum['ELO_gt'] = []
                for answer in szFreq_datum['answers']['text']:
                    combined_datum['ELO_gt'].append(answer)
                if not combined_datum['ELO_gt']:
                    combined_datum['ELO_gt'] = ""
            break
    
    #if this is a new document and passage, make a new dictionary
    #then, add the current question to it
    if not has_entry:
        #initialize this entry's dictionary
        combined_datum = {'doc_name':szFreq_datum['title'],
                          'passage':szFreq_datum['context']}
        
        #add the current question answer to it
        if szFreq_datum['question'] == 'How often does the patient have events?':
            combined_datum['PQF_q'] = szFreq_datum['question']
            combined_datum['PQF_id'] = szFreq_datum['id'] 
            combined_datum['PQF_gt'] = []
            for answer in szFreq_datum['answers']['text']:
                combined_datum['PQF_gt'].append(answer)
            if not combined_datum['PQF_gt']:
                combined_datum['PQF_gt'] = ""
        else:
            combined_datum['ELO_q'] = szFreq_datum['question']
            combined_datum['ELO_id'] = szFreq_datum['id'] 
            combined_datum['ELO_gt'] = []
            for answer in szFreq_datum['answers']['text']:
                combined_datum['ELO_gt'].append(answer)
            if not combined_datum['ELO_gt']:
                combined_datum['ELO_gt'] = ""
        
        #add the entry into the list
        combined_data.append(combined_datum)
        
                        
#iterate through the szFreq predictions and build up the combined list
for szFreq_pred_id in szFreq_predictions:
    #find which dict in combined_data this question belongs to and store its answer
    for datum in combined_data:
        if szFreq_pred_id == datum['PQF_id']:
            datum['PQF_pred'] = szFreq_predictions[szFreq_pred_id]
            break
        elif szFreq_pred_id == datum['ELO_id']:
            datum['ELO_pred'] = szFreq_predictions[szFreq_pred_id]
            break

#iterate through the hasSz dataset. note that the indicies of hasSz dataset and predictions
#match
for i in range(len(hasSz_dataset)):
    #find which dict in combined_data this hasSz question belongs to and add the question, prediction ground truth answer
    for combined_datum in combined_data:
        if combined_datum['passage'] == hasSz_dataset[i]['passage']:
            combined_datum['hasSz_q'] = hasSz_dataset[i]['question']
            combined_datum['hasSz_gt'] = one_hot_reversed[hasSz_dataset[i]['label']]
            combined_datum['hasSz_pred'] = one_hot_reversed[hasSz_predictions_labels[i]]
            break
        
with open(args.output_path, 'w') as f:
    for datum in combined_data: 
        json.dump(datum, f)

        f.write('\n')
