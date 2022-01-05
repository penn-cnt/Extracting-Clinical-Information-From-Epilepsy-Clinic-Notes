import pandas as pd
import json
import numpy as np
import csv
import os
import itertools
import string
import annotation_utils as anno_utils
from difflib import SequenceMatcher
from functools import reduce

#what are the local directorty of the annotation groups
proj_dir = './'

#where are the QA test datasets
squad_test_save_dir = 'QA_Datasets/text_extraction_test.json'
boolqa_test_save_dir = 'QA_Datasets/classification_test.json'

#where do we want to save the "predictions"
prediction_dir = 'human_baseline_predictions/'

#create a 1-hot-encoding dictionary, assuming True, False, no-answer. 
#Yes and No map equally to true and false. 
#unspecified or missing answer maps no-answer
one_hot = {"true":1, "false":0, "no-answer":2, 'yes':1, 'no':0, '-1':2, 'unspecified':2}

#read the szFreq_test and hasSz_test datasets
with open(squad_test_save_dir, 'r') as f:
    szFreq_test = json.load(f)['data']
with open(boolqa_test_save_dir, 'r') as f:
    hasSz_test = [json.loads(qa) for qa in f.read().splitlines()]
    
#iterate through the dataset and get the titles of the documents.
hasSz_test_titles = [qa['title'] for qa in hasSz_test]
szFreq_test_titles = [qa['title'] for qa in szFreq_test]

#find the list of titles of the documents with both a szFreq and hasSz (should be 100% intersection between the two)
testset_titles = list(set(hasSz_test_titles) & set(szFreq_test_titles))

#iterate through the annotation groups and collect the annotators
annotator_info = {}
annotator_to_index = {}
num_annotators = 0

#iterate through annotation groups
for directory in os.listdir(proj_dir):
    #skip if it isn't a group's directory
    if "_annotations" not in directory:
        continue
        
    #get the annotated documents from this annotation group
    anno_dir = proj_dir+"/"+directory+"/annotation"
    anno_docs = os.listdir(anno_dir)
    
    for doc in anno_docs:
        doc_dir = anno_dir+"/"+doc
        anno_files = os.listdir(doc_dir)

        #iterate through annotations and get basic info of annotators
        for anno in anno_files:

            username = anno[:-4]

            #add a new annotator if necessary
            if username not in annotator_info:
                annotator_info[username] = 0
                annotator_to_index[username] = num_annotators
                num_annotators += 1

            annotator_info[username] += 1

#convert between annotator and their index
index_to_annotator = {v:k for k,v in annotator_to_index.items()}

#create annotators
all_annotators = []
for i in range(num_annotators):
    all_annotators.append(anno_utils.Annotator(index_to_annotator[i], i))


base_cols = ['Sen-Tok', 'Beg-End', 'Token']
all_documents = []
#iterate through annotation groups
for directory in os.listdir(proj_dir):
    #skip if it isn't a group's directory
    if "_annotations" not in directory:
        continue
        
    #get the annotated documents from this annotation group
    anno_dir = proj_dir+"/"+directory+"/annotation"
    anno_docs = os.listdir(anno_dir)
    
    #for each annotated document in this annotation group
    for doc in anno_docs:
        
        #skip this document if it wasn't in the test set
        if doc not in testset_titles:
            continue
        
        #find the document to be annotated and get the annotators
        doc_dir = anno_dir+"/"+doc
        anno_files = os.listdir(doc_dir)
        
        #create a new Document
        new_doc = anno_utils.Document(doc)
        
        #iterate through annotations for this document and get annotation data
        annotations = {}
        for anno in anno_files:
            filename = doc_dir+"/"+anno
            anno_doc = pd.read_csv(filename, comment='#', sep='\t+',\
                         header=None, quotechar='"', engine='python', \
                         names=anno_utils.GetHeaders(filename), index_col=None)
            if len(anno_doc.columns) < 4:
                continue
            if 'HasSeizures' not in anno_doc.columns:
                anno_doc['HasSeizures'] = "_"
            if 'SeizureFrequency' not in anno_doc.columns:
                anno_doc['SeizureFrequency'] = "_"
            if 'TypeofSeizure' not in anno_doc.columns:
                anno_doc['TypeofSeizure'] = "_"
            if 'referenceRelation' in anno_doc.columns:
                anno_doc = anno_doc.drop('referenceRelation', axis=1)
            if 'referenceType' in anno_doc.columns:
                anno_doc = anno_doc.drop('referenceType', axis=1)
            #rename columns based off of annotator index
            anno_doc = anno_doc.rename(columns={'HasSeizures':'HasSeizures_'+str(annotator_to_index[anno[:-4]]), \
                                               'SeizureFrequency':'SeizureFrequency_'+str(annotator_to_index[anno[:-4]]), \
                                               'TypeofSeizure':'TypeofSeizure_'+str(annotator_to_index[anno[:-4]])})        
            annotations[annotator_to_index[anno[:-4]]] = anno_doc
            all_annotators[annotator_to_index[anno[:-4]]].add_document(new_doc)
            
        #skip if no annotations have been performed
        if not bool(annotations):
            print("Missing Annotations: " + str(doc))
            continue;        
            
        #combine annotations into a single table
        annotations = reduce(lambda df1, df2: pd.merge(df1, df2, how='outer', 
                                                       on=['Sen-Tok', 'Beg-End', 'Token']), annotations.values())
        
        #replace empty values
        annotations = annotations.fillna('_')
        #replace incomplete annotations with blanks
        annotations = annotations.replace(to_replace=r'\*.+|\*', value='_', regex=True)    

        #add text to the new Document
        new_doc.set_text(annotations[['Beg-End','Token']])
        #add Document to document container
        all_documents.append(new_doc)

        #process annotations
        anno_utils.collect_annotations(annotations.drop(base_cols,1), new_doc, all_annotators)

#Generate predictions on the classification question
pred_counters = {}
for annotator in all_annotators:
    pred_counters[annotator_to_index[annotator.name]] = 0

for qa in hasSz_test:
    #find the corresponding document
    doc = anno_utils.get_doc_by_name(all_documents, qa['title'])
    
    #create a dictionary that keeps track of hasSz annotations for each annotator
    #if they have more than one hasSz annotation, then skip it for that annotator
    hasSz_annotations = {}
    
    #for each annotation in this document
    for anno in doc.annotations:
        #check if the annotation is of the HasSeizures layer
        if anno.layer != 'HasSeizures':
            continue
        
        #if this annotator already had an annotation for this qa, then skip them
        if annotator_to_index[anno.Annotator.name] in hasSz_annotations:
            print("Multiple hasSz found for " + anno.Annotator.name + " in document " + doc.name)
            hasSz_annotations[annotator_to_index[anno.Annotator.name]] = None
        else:
            hasSz_annotations[annotator_to_index[anno.Annotator.name]] = one_hot[anno.get_raw_value().lower()]
            
    #for each HasSz annotator, get their prediction
    for annotator_idx in hasSz_annotations:
        if hasSz_annotations[annotator_idx] is None:
            continue
        else:
            pred_vector = [0, 0, 0]
            pred_vector[hasSz_annotations[annotator_idx]] = 1   
            pred_string = "["+" ".join(map(str, pred_vector))+"]"
            with open(prediction_dir+"classification_annotator_"+str(annotator_idx)+"_predictions.tsv", 'a') as f:
                if pred_counters[annotator_idx] == 0:
                    f.write('Predictions\tTrue_Label')
                    f.write('\n')
                f.write(pred_string + "\t" + str(one_hot[str(qa['answer']).lower()]))
                f.write('\n')
                pred_counters[annotator_idx] += 1

#generate predictions on the text extraction questions
#create a dictionary of dictionaries. Each dictionary will be the annotator's predictions in the format {question_id:predicted_text}. The overall structure is {annotator_id:predictions}
all_predictions = {}
for annotator in all_annotators:
    all_predictions[annotator_to_index[annotator.name]] = {}

for qa in szFreq_test:
    #find the corresponding document
    doc = anno_utils.get_doc_by_name(all_documents, qa['title'])
    
    #what's the PQF question id?
    pqf_id = qa['paragraphs'][0]['qas'][0]['id'] if qa['paragraphs'][0]['qas'][0]['question'] == "How often does the patient have events?" else qa['paragraphs'][0]['qas'][1]['id']
    #what's the ELO question id?
    elo_id = qa['paragraphs'][0]['qas'][0]['id'] if qa['paragraphs'][0]['qas'][0]['question'] == "When was the patient's last event?" else qa['paragraphs'][0]['qas'][1]['id']
    
    #create dictionaries that keeps track of szFreq annotations for each annotator
    PQF_annotations = {}
    ELO_annotations = {}
    
    #for each annotation in this document
    for anno in doc.annotations:
        #check if the annotation is of the SeizureFrequency layer
        if anno.layer != 'SeizureFrequency':
            continue
            
        if anno.get_raw_value() == 'PositiveQuantitativeFrequency':
            #update the annotator's annotation entry
            if annotator_to_index[anno.Annotator.name] in PQF_annotations:
                PQF_annotations[annotator_to_index[anno.Annotator.name]].append(anno.get_text())
            else:
                PQF_annotations[annotator_to_index[anno.Annotator.name]] = []
                PQF_annotations[annotator_to_index[anno.Annotator.name]].append(anno.get_text())
        else:
            #update the annotator's annotation entry
            if annotator_to_index[anno.Annotator.name] in ELO_annotations:
                ELO_annotations[annotator_to_index[anno.Annotator.name]].append(anno.get_text())
            else:
                ELO_annotations[annotator_to_index[anno.Annotator.name]] = []
                ELO_annotations[annotator_to_index[anno.Annotator.name]].append(anno.get_text())
            
                    
    #for each PQF annotator, get their prediction
    for annotator_idx in PQF_annotations:
        
        #dictionary to hold the annotator's possible predictions
        #structure: {answer_text:max_ratio}
        possible_predictions = {}
        
        #for each PQF annotation,
        for anno in PQF_annotations[annotator_idx]:
            
            #for each ground truth question
            for question in qa['paragraphs'][0]['qas']:
                
                #check if it's a PQF question
                if question['question'] == "When was the patient's last event?":
                    continue
                
                #get the highest similarity score between this annotation and the ground-truth answers
                possible_predictions[anno] = SequenceMatcher(None, anno, "").ratio() if question['is_impossible'] else max([SequenceMatcher(None, anno, answer['text']).ratio() for answer in question['answers']])
                
        #pick the annotation with the best similarity to an answer as the final prediction and add it to the all_predictors dictionary
        all_predictions[annotator_idx][pqf_id] = max(possible_predictions, key = possible_predictions.get)
        
    #for each ELO annotator, get their prediction
    for annotator_idx in ELO_annotations:
        
        #dictionary to hold the annotator's possible predictions
        #structure: {answer_text:max_ratio}
        possible_predictions = {}
        
        #for each PQF annotation,
        for anno in ELO_annotations[annotator_idx]:
            
            #for each ground truth question
            for question in qa['paragraphs'][0]['qas']:
                
                #check if it's a ELO question
                if question['question'] == "How often does the patient have events?":
                    continue
                
                #get the highest similarity score between this annotation and the ground-truth answers
                possible_predictions[anno] = SequenceMatcher(None, anno, "").ratio() if question['is_impossible'] else max([SequenceMatcher(None, anno, answer['text']).ratio() for answer in question['answers']])
                
        #pick the annotation with the best similarity to an answer as the final prediction and add it to the all_predictors dictionary
        all_predictions[annotator_idx][elo_id] = max(possible_predictions, key = possible_predictions.get)
        
    #add in 'no answer's if an annotation doesn't exist
    for annotator in doc.annotators:
        if pqf_id not in all_predictions[annotator_to_index[annotator.name]]:
            all_predictions[annotator_to_index[annotator.name]][pqf_id] = ""
        if elo_id not in all_predictions[annotator_to_index[annotator.name]]:
            all_predictions[annotator_to_index[annotator.name]][elo_id] = ""

#write all classification predictions to file
for annotator_idx in all_predictions:
    with open(prediction_dir+"text_extraction_annotator_"+str(annotator_idx)+"_predictions.json", 'w') as f:
        json.dump(all_predictions[annotator_idx], f)
