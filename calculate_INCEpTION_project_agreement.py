import pandas as pd
import numpy as np
import csv
import os
import annotation_utils as anno_utils
from functools import reduce

#what is the minimum number of annotations to be considered for agreement?
min_annotations = 50

#what directory (annotation, curation, combined) are we analyzing in?
analysis_dir = "combined"

#what is the exported project's local directory?
proj_dir = 'Group_1_annotations'

#list all document subdirectories. Inside each directory will be the tsv files for each annotator that has performed an annotation
anno_docs = os.listdir(proj_dir+"/" + analysis_dir)
num_files = len(anno_docs)

#tabulate all annotators, get the number of documents each have done
annotator_info = {}
annotator_to_index = {}
num_annotators = 0

#iterate through documents and get annotations
for doc in anno_docs:
    doc_dir = proj_dir+"/"+analysis_dir+"/"+doc
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

#maps index to annotator name
index_to_annotator = {v:k for k,v in annotator_to_index.items()}
        
print(annotator_info)
print(annotator_to_index)
print(index_to_annotator)

#create annotator objects
all_annotators = []
for i in range(num_annotators):
    all_annotators.append(anno_utils.Annotator(index_to_annotator[i], i))


base_cols = ['Sen-Tok', 'Beg-End', 'Token']
all_documents = []
#collect annotations for each Document
for doc in anno_docs:
    #find the document to be annotated and the annotators
    doc_dir = proj_dir+"/"+analysis_dir+"/"+doc
    anno_files = os.listdir(doc_dir)
    
    #create a new Document
    new_doc = anno_utils.Document(doc)
    
    #iterate through annotations for this document and get their annotation data
    annotations = {}
    for anno in anno_files:
        #skip if the annotator has less than complete annotations
        if annotator_info[anno[:-4]] < min_annotations:
            continue
        
        #read the annotations and standardize the headers    
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

#for each annotator, compare them to the others and identify overlapping annotations
for i in range(len(all_annotators)):
    for j in range(i+1, len(all_annotators)):
        
        #iterate through the annotations of each annotator
        for anno_1 in range(len(all_annotators[i].annotations)):
            for anno_2 in range(len(all_annotators[j].annotations)):
                if not all_annotators[i].annotations[anno_1].check_overlap(all_annotators[j].annotations[anno_2]):
                    continue
                all_annotators[i].annotations[anno_1].add_overlap(all_annotators[j].annotations[anno_2])
                all_annotators[j].annotations[anno_2].add_overlap(all_annotators[i].annotations[anno_1])
                all_annotators[i].annotations[anno_1].info()
                all_annotators[j].annotations[anno_2].info()

#print out the pairwise agreement between annotators of this annotation group
for i in range(len(all_annotators)):
    for j in range(i+1, len(all_annotators)):   
        if not all_annotators[i].annotations or not all_annotators[j].annotations:
            continue
        
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print("Calculating agreement between " + all_annotators[i].name + " and " + all_annotators[j].name)
        iaa = anno_utils.Agreement(all_annotators[i], all_annotators[j])
        iaa.calc_simple_agreement()
        
        print()
        print("HasSz Agreement: ")
        print("Cohen's Kappa: " + str(iaa.calc_cohen_kappa('HasSeizures')))
        print()
        
        print("SzFreq Agreement: ")
        print("Cohen's Kappa: " + str(iaa.calc_cohen_kappa('SeizureFrequency'))) 
        print("F1 Overlap: " + str(iaa.calc_average_F1_overlap('SeizureFrequency')))
        print()
        
        print("SzType Agreement: ")
        print("Cohen's Kappa: " + str(iaa.calc_cohen_kappa('TypeofSeizure'))) 
        print("F1 Overlap: " + str(iaa.calc_average_F1_overlap('TypeofSeizure')))
        print()
        
        iaa.info()
    
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")                    
