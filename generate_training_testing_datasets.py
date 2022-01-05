import pandas as pd
import json
import numpy as np
import csv
import os
import annotation_utils as anno_utils
from functools import reduce
from sklearn.model_selection import train_test_split

#what is the exported project's local directory?
proj_dir = 'combined_curation'

#where are we saving the QA datasets
squad_train_save_dir = 'QA_Datasets/text_extraction_train'
boolqa_train_save_dir = 'QA_Datasets/classification_train'
squad_test_save_dir = 'QA_Datasets/text_extraction_test'
boolqa_test_save_dir = 'QA_Datasets/classification_test'

#train decimal in the train test split
train_split = 0.7

#what partitions are we saving on for the training data?
#this will be used for testing number of examples vs. validation performance
save_percent = 0.1 #save every 10%

#What are our questions?
hasSz_Qs = "Has the patient had recent events?"
pqf_Qs = ["How often does the patient have events?"]
elo_Qs = ["When was the patient's last event?"]
squad_questions = pqf_Qs + elo_Qs

#list all annotation directories. Inside each directory will be the tsv files for each annotator that has performed an annotation
anno_docs = os.listdir(proj_dir)
num_files = len(anno_docs)

#tabulate all annotators, get the number of documents each has done
annotator_info = {}
annotator_to_index = {}
num_annotators = 0

#iterate through documents and get annotations
for doc in anno_docs:
    doc_dir = proj_dir+"/"+doc
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

index_to_annotator = {v:k for k,v in annotator_to_index.items()}
        
print(annotator_info)
print(annotator_to_index)
print(index_to_annotator)

#create annotators
all_annotators = []
for i in range(num_annotators):
    all_annotators.append(anno_utils.Annotator(index_to_annotator[i], i))

base_cols = ['Sen-Tok', 'Beg-End', 'Token']
all_documents = []

#collect annotations for each Document
for doc in anno_docs:
    #find the document to be annotated and the annotators
    doc_dir = proj_dir+"/"+doc
    anno_files = os.listdir(doc_dir)
    
    #create a new Document
    new_doc = anno_utils.Document(doc)
    
    #iterate through annotations for this document and get their annotation data
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


#perform train-test split
train_docs, test_docs = train_test_split(all_documents, train_size=train_split)
print(len(train_docs))
print(len(test_docs))

#Save the training data
#Initialize the SQuAD dictionary and BoolQA list
squad = {'data':[], 'version':'v2.0'}
bool_qs = []
squad_id = 0

doc_counter = 0
fraction_counter = 1

#for each document
for doc in train_docs:
    squad_qas = []
    
    #append the date each note was written to the front of the note.
    doc_date = doc.name.split("_")[2]
    doc_text_append = "This note was written on " + doc_date + ". "
    doc_text = doc_text_append + doc.get_raw_text()

    #add all potential squad questions as blanks
    for q in squad_questions:
        squad_qas.append({'question':q,
                   'is_impossible':True,
                   'id':str(squad_id),
                   'answers':[]})
        squad_id += 1
        
    #add all potential boolqa questions as blanks
    bool_qa = {'answer':-1,
                'passage':doc_text,
                'question':hasSz_Qs,
                'title':doc.name}
        
    #for each annotation, add annotations as the answer to each question
    for annotation in doc.annotations:
        #if the annotation is for HasSz,
        if annotation.layer == "HasSeizures":
            #add the answer to the question
            bool_qa['answer'] = annotation.get_raw_value() if annotation.get_raw_value() != 'Unspecified' else 'no-answer'
            
        elif annotation.layer == 'SeizureFrequency':
            if annotation.get_raw_value() == 'ExplicitLastOccurrence':
                for qa in squad_qas:
                    if qa['question'] in elo_Qs:
                        qa['is_impossible'] = False
                        qa['answers'].append({
                            'text':annotation.get_text(),
                            'answer_start':doc_text.find(annotation.get_text())
                        })
            else:
                for qa in squad_qas:
                    if qa['question'] in pqf_Qs:
                        qa['is_impossible'] = False
                        qa['answers'].append({
                            'text':annotation.get_text(),
                            'answer_start':doc_text.find(annotation.get_text())
                        })
                 
    #condense the qas into the appropriate SQuAD dictionary
    doc_entry = {'title':doc.name, 
                 'paragraphs':[
                     {'context':doc_text, 
                      'qas':squad_qas}
                 ]
                }
    
    
    #add this document to the whole squad dataset
    squad['data'].append(doc_entry)
    
    #add this document to the boolqa dataset
    bool_qs.append(bool_qa)
    
    #write datasets to files
    doc_counter += 1
    if doc_counter == int(save_percent * fraction_counter * len(train_docs)):
        with open(squad_train_save_dir+'_'+str(round(save_percent * fraction_counter, 1))+'.json', 'w') as f:
            json.dump(squad, f)
        with open(boolqa_train_save_dir+'_'+str(round(save_percent * fraction_counter, 1))+'.json', 'w') as f:
            for qa in bool_qs:
                json.dump(qa, f)
                f.write('\n')
        fraction_counter += 1


#Save the validation  data
#Initialize the SQuAD dictionary and BoolQA list
squad = {'data':[], 'version':'v2.0'}
bool_qs = []
squad_id = 0

#for each document
for doc in test_docs:
    squad_qas = []
    
    #append the date each note was written to the front of the note.
    doc_date = doc.name.split("_")[2]
    doc_text_append = "This note was written on " + doc_date + ". "
    doc_text = doc_text_append + doc.get_raw_text()

    #add all potential squad questions as blanks
    for q in squad_questions:
        squad_qas.append({'question':q,
                   'is_impossible':True,
                   'id':str(squad_id),
                   'answers':[]})
        squad_id += 1
        
    #add all potential boolqa questions as blanks
    bool_qa = {'answer':-1,
                'passage':doc_text,
                'question':hasSz_Qs,
                'title':doc.name}
        
    #for each annotation, add annotations as the answer to each question
    for annotation in doc.annotations:
        #if the annotation is for HasSz,
        if annotation.layer == "HasSeizures":
            #add the answer to the question
            bool_qa['answer'] = annotation.get_raw_value() if annotation.get_raw_value() != 'Unspecified' else 'no-answer'
            
        elif annotation.layer == 'SeizureFrequency':
            if annotation.get_raw_value() == 'ExplicitLastOccurrence':
                for qa in squad_qas:
                    if qa['question'] in elo_Qs:
                        qa['is_impossible'] = False
                        qa['answers'].append({
                            'text':annotation.get_text(),
                            'answer_start':doc_text.find(annotation.get_text())
                        })
            else:
                for qa in squad_qas:
                    if qa['question'] in pqf_Qs:
                        qa['is_impossible'] = False
                        qa['answers'].append({
                            'text':annotation.get_text(),
                            'answer_start':doc_text.find(annotation.get_text())
                        })
                 
    #condense the qas into the appropriate SQuAD dictionary
    doc_entry = {'title':doc.name, 
                 'paragraphs':[
                     {'context':doc_text, 
                      'qas':squad_qas}
                 ]
                }
    
    
    #add this document to the whole squad dataset
    squad['data'].append(doc_entry)
    
    #add this document to the boolqa dataset
    bool_qs.append(bool_qa)

#write to the test files    
with open(squad_test_save_dir+'.json', 'w') as f:
    json.dump(squad, f)
with open(boolqa_test_save_dir+'.json', 'w') as f:
    for qa in bool_qs:
        json.dump(qa, f)
        f.write('\n')
