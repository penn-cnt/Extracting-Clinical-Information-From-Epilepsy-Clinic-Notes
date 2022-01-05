"""
Extracts hpi and interval history from a table of progress notes. 
The table contains column heads of pat_id, visit_date, note_author, note_text, and others. 
Each row of the table represents a single medical note.
We do not include this table out of PHI concerns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import copy
import random
from fuzzywuzzy import fuzz
import os

#set true if you want to generate a uniform distribution of providers
uniform_dist = True

#What is the main directory of the annotation project documents (subdirectories will be created automatically)
main_directory = 'Annotation_Project_Groups'
num_subdirectories = 5 #five annotation groups

#number of extracted paragraphs per subdirectory
num_extracts = 200

#Make subdirectories
for i in range(num_subdirectories):
    os.makedirs(main_directory+"/"+"Group_"+str(i), exist_ok=True)

#load the progress notes file
c_notes = pd.read_csv("Progress_Notes.csv")[['pat_id', 'visit_date', 'note_author', 'note_text']]

#inclusion (whitelist) and exclusion criteria
#author names redacted for privacy reasons.
whitelist_authors = ["specialist_1, specialist_2, specialist_3, specialist_4, specialist_5, specialist_6, specialist_7, specialist_8"]
addendum_string = "I saw and evaluated  on  and reviewed 's notes. I agree with the history, physical exam, and medical decision making with following additions/exceptions/observations"
whitelist_regex = r"(?im)^(\bHPI\b|\bHistory of Present Illness\b|\bInterval History\b)"
blacklist_regex = r"(?im)(\b(Past |Prior )?((Social|Surgical|Family|Medical|Psychiatric|Seizure|Disease|Epilepsy) )?History\b|\bSemiology\b|\bLab|\bExam|\bDiagnostic|\bImpression|\bPlan\b|\bPE\b|\bRisk Factor|\bMedications)"

#new lines in the notes are represented as two spaces ("  ")
splitter = "  "

#maximum sequence length. We estimate average 3 characters per token, although this is dependent on the tokenizer
max_length = 3*512 - 30

#keep track of documents and paragraphs
extracted_paragraphs = []
num_possible_documents = 0
num_docs_one_head = 0
num_docs_multi_head = 0

#for each document (progress note)
for c_index, c_row in c_notes.iterrows():

    #simple counter and progress tracker
    if c_index % 10000 == 0:
        print("Starting " + str(c_index))
    
    #skip notes that aren't written by the whitelist authors
    if str(c_row['note_author']).lower() not in whitelist_authors:
        continue;
    
    #split the document into its lines
    sentences = c_row['note_text'].strip('"').split(splitter)
    
    #skip notes that are less than 3 lines long
    if len(sentences) < 3:
        continue
    
    #check for attending addendum characteristics in the first three lines and skip if it exists
    isAttendingAddendum = False
    for i in range(3):
        if ("Attending Addendum" in sentences[i]) or (fuzz.ratio(addendum_string, sentences[i]) >= 75):
            continue
        
    #Dictionary to store relevant information of the document
    document = {}
    document['filename'] = f"{c_row['pat_id']}_{c_row['note_author']}_{c_row['visit_date'][:10]}"
    document['note_author'] = c_row['note_author']
    num_possible_documents += 1
        
    #scan through each line and find indices where it contains a desired header
    whitelist_indices = []
    blacklist_indices = []
    header_indices = []
    for i in range(len(sentences)):
        substr = sentences[i].strip()[:30]
        if re.search(whitelist_regex, substr):
            whitelist_indices.append(i)
            header_indices.append(i)
        elif re.search(blacklist_regex, substr):
            blacklist_indices.append(i)
            header_indices.append(i)
    header_indices.append(-1)        

    #if no whitelisted header is found, skip this note
    if len(whitelist_indices) < 1:
        continue
    
    num_headers = 0
    extract_counter = 0
    
    #for each whitelisted header, extract its text until the next (white or blacklisted) header starts, or until max_length number of characters
    for i in range(1, len(header_indices)):
        if header_indices[i-1] in blacklist_indices:
            continue
        elif header_indices[i] == -1:
            doc_text = "".join([line.strip() + "\n" for line in sentences[header_indices[i-1]:] if line != ""])[:max_length]
            if len(doc_text) > 250:
                document[extract_counter] = doc_text
                extract_counter += 1
        else:
            doc_text = "".join([line.strip() + "\n" for line in sentences[header_indices[i-1]:header_indices[i]] if line != ""])[:max_length]
            if len(doc_text) > 250:
                document[extract_counter] = doc_text
                extract_counter += 1
        num_headers += 1
        
    if num_headers == 1:
        num_docs_one_head += 1
    elif num_headers > 1:
        num_docs_multi_head += 1
    
    #skip if there were no paragraphs extracted    
    if len(document) > 2:
        extracted_paragraphs.append(document)      

#randomize the order of documents
random.shuffle(extracted_paragraphs)
extracted_paragraphs = pd.DataFrame(extracted_paragraphs)

#print some statistics - number of docs with one paragraph, and number of docs with more than one head
print(num_docs_one_head / num_possible_documents)
print(num_docs_multi_head / num_possible_documents)

#for counting number of notes for each author to use in a uniform distribution
num_notes_per_author = dict.fromkeys(whitelist_authors, 0)
max_notes_per_author = dict.fromkeys(whitelist_authors, 0)

#find how many notes each provider has authored
for author in max_notes_per_author:
    max_notes_per_author[author] = len(extracted_paragraphs.loc[extracted_paragraphs['note_author'] == author.upper()])

#find how many notes each author can write for a uniform distribution
num_notes = 0
while num_notes < num_extracts*num_subdirectories:
    for author in whitelist_authors:
        if num_notes_per_author[author] >= max_notes_per_author[author]:
            continue
            
        num_notes_per_author[author] += 1
        num_notes += 1
        
        if num_notes >= num_extracts*num_subdirectories:
            break

#how many notes each author can write in each subdirectory
for author in num_notes_per_author:
    num_notes_per_author[author] = int(num_notes_per_author[author]/num_subdirectories)
    
#for each subdirectory
for subdir in os.listdir(main_directory):
    
    #Skip irrelevant subfolders
    if 'Group' not in subdir:
        continue
    
    #initialize an empty dataframe
    extracted_subset = pd.DataFrame()
    for head in extracted_paragraphs.columns:
        extracted_subset[head] = None
    
    #create the subset of notes that we want to take passages from
    if uniform_dist:
        for author in num_notes_per_author:
            extracted_subset = extracted_subset.append(extracted_paragraphs.loc[extracted_paragraphs['note_author'] == author.upper()].sample(num_notes_per_author[author]))
    else:
        extracted_subset = extracted_paragraphs.sample(num_extracts)
    extracted_paragraphs = extracted_paragraphs.drop(extracted_subset.index)
    
    #for each note, save the text in the subdirectory
    for e_index, e_row in extracted_subset.iterrows():
        #randomly sample one paragraph that isn't NaN
        sample = e_row[e_row.notna()][2:].sample(1)
        sample_idx = sample.index.tolist()[0]
        sample_text = sample.values[0]
        with open(main_directory+"/"+subdir+"/"+e_row['filename']+'_'+str(sample_idx)+".txt", 'w') as f:
                f.write(sample_text)
