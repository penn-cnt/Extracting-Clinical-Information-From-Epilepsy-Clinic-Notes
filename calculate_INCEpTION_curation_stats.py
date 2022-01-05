import pandas as pd
import numpy as np
import re
import csv
import os
from annotation_utils import GetHeaders

#what is the exported project's local directory?
proj_dir = 'combined_curation'
anno_docs = os.listdir(proj_dir)

#keep track of statistics for the classification and text extraction tasks
#Positive Quantitative Frequency (PQF) refers to the seizure frequency extraction task
#Explicit Last Occurrence (ELO) refers to the date of most recent seizure task
classification_stats = {"Yes":0, "No":0, "Unspecified":0}
text_extraction_stats = {"None":0, 
                "Positive Quantitative Frequency":0, 
                'Explicit Last Occurrence':0,
                'Both':0,
                'Distinct_PQF':0,
                'Distinct_ELO':0,
                'Total_PQF':0,
                'Total_ELO':0               }

for doc in anno_docs:
    doc_dir = proj_dir+"/"+doc
    filename = doc_dir+"/CURATION_USER.tsv"
    
    #read the annotation document
    anno_doc = pd.read_csv(filename, comment='#', sep='\t+',\
                 header=None, quotechar='"', engine='python', \
                 names=GetHeaders(filename), index_col=None)
    if len(anno_doc.columns) < 4:
        print("Error: No annotations in document "+doc)
        continue
    
    #standardize the headers
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
        
    #replace empty values
    anno_doc = anno_doc.fillna('_')
    #replace incomplete annotations with blanks
    anno_doc = anno_doc.replace(to_replace=r'\*.+', value='_', regex=True)    
    
    #calculate classification stats
    classification_anno = [re.sub(r'\[[^()]*\]',"",anno) for anno in anno_doc['HasSeizures'].unique() if anno != '_']
    if len(classification_anno) != 1:
        print("Error: No classification in document "+ doc+", Defaulting to Unspecified")
        classification_stats['Unspecified'] += 1
    else:
        classification_stats[classification_anno[0]]+=1
    
    #calculate general text extraction stats
    text_extraction_anno = [anno for anno in anno_doc['SeizureFrequency'].unique() if anno != "_"]
    text_extraction_anno_values = list(set([re.sub(r'\[[^()]*\]',"",anno) for anno in text_extraction_anno]))
    if not text_extraction_anno_values:
        text_extraction_stats['None'] += 1
    elif len(text_extraction_anno_values) == 2:
        text_extraction_stats['Both']+=1
    elif len(text_extraction_anno_values) > 2:
        print("Error: text extraction problem in document "+ doc)
    else:
        text_extraction_stats[text_extraction_anno_values[0]] += 1
    
    #calculate specific text_extraction stats
    num_pqf = 0
    num_elo = 0
    for anno in text_extraction_anno:
        if "Positive" in anno:
            text_extraction_stats['Total_PQF'] += 1
            num_pqf += 1
        elif "Explicit" in anno:
            text_extraction_stats['Total_ELO'] += 1
            num_elo += 1
    if num_pqf > 1:
        text_extraction_stats['Distinct_PQF'] += 1
    if num_elo > 1:
        text_extraction_stats['Distinct_ELO'] += 1
        
print("Number of notes with classification = Yes: " + str(classification_stats['Yes']) + " ("+str(classification_stats['Yes']/sum(classification_stats.values())*100)+"%)")
print("Number of notes with classification = No: " + str(classification_stats['No']) + " ("+str(classification_stats['No']/sum(classification_stats.values())*100)+"%)")
print("Number of notes with classification = Unspecified: " + str(classification_stats['Unspecified']) + " ("+str(classification_stats['Unspecified']/sum(classification_stats.values())*100)+"%)")
print("")
print("Number of notes with only text extraction = PQF: " + str(text_extraction_stats['Positive Quantitative Frequency']) + " ("+str(text_extraction_stats['Positive Quantitative Frequency']/sum(classification_stats.values())*100)+"%)")
print("Number of notes with only text extraction = ELO: " + str(text_extraction_stats['Explicit Last Occurrence']) + " ("+str(text_extraction_stats['Explicit Last Occurrence']/sum(classification_stats.values())*100)+"%)")
print("Number of notes with at least text extraction = PQF: " + str(text_extraction_stats['Positive Quantitative Frequency']+text_extraction_stats['Both']) + " ("+str((text_extraction_stats['Positive Quantitative Frequency']+text_extraction_stats['Both'])/sum(classification_stats.values())*100)+"%)")
print("Number of notes with at least text extraction = ELO: " + str(text_extraction_stats['Explicit Last Occurrence']+text_extraction_stats['Both']) + " ("+str((text_extraction_stats['Explicit Last Occurrence']+text_extraction_stats['Both'])/sum(classification_stats.values())*100)+"%)")
print("Number of notes with both text extraction = PQF and text_extraction = ELO: " + str(text_extraction_stats['Both']) + " ("+str(text_extraction_stats['Both']/sum(classification_stats.values())*100)+"%)")
print("Number of notes with some text extraction annotations: " + str(text_extraction_stats['Positive Quantitative Frequency']+text_extraction_stats['Explicit Last Occurrence']+text_extraction_stats['Both']) + \
      " ("+str((text_extraction_stats['Positive Quantitative Frequency']+text_extraction_stats['Explicit Last Occurrence']+text_extraction_stats['Both']) / sum(classification_stats.values())*100)+"%)")
print("")
print("Number of notes with two or more PQF: " + str(text_extraction_stats['Distinct_PQF']) + " ("+str(text_extraction_stats['Distinct_PQF']/sum(classification_stats.values())*100)+"%)")
print("Number of notes with two or more PQF: " + str(text_extraction_stats['Distinct_ELO']) + " ("+str(text_extraction_stats['Distinct_ELO']/sum(classification_stats.values())*100)+"%)")
print("Total number of ELO annotations: "+str(text_extraction_stats['Total_ELO']))
print("Total number of PQF annotations: "+str(text_extraction_stats['Total_PQF']))
