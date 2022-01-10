from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher
from datetime import datetime
import seaborn as sns
sns.set_theme(style='ticks')

def get_paragraph(whitelist_regex, blacklist_regex, note_text, note_author, pat_id, visit_date, splitter="  ", max_length=(3*512-30)):
    #split the document into lines
    sentences = note_text.strip('"').split(splitter)
    
    #skip notes that are less than 3 lines long
    if len(sentences) < 3:
        return None
        
    #Dictionary to store relevant information of the document
    document = {}
    document['filename'] = f"{pat_id}_{note_author}_{visit_date[:10]}"
    document['note_author'] = note_author
    
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
        return None
    
    #extract the paragraphs starting from a whitelist header until the next white or blacklisted header. 
    extract_counter = 0
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
    
    #return if there are any extracted paragraphs    
    if len(document) > 2:
        return document


#read the progress notes and set up inclusion and exclusion criteria
p_notes = pd.read_csv("Progress_Notes.csv")[['pat_id', 'visit_date', 'note_author', 'note_text']]
#author names redacted for privacy reasons.
whitelist_authors = ["specialist_1, specialist_2, specialist_3, specialist_4, specialist_5, specialist_6, specialist_7, specialist_8"]
addendum_string = "I saw and evaluated  on  and reviewed 's notes. I agree with the history, physical exam, and medical decision making with following additions/exceptions/observations"
whitelist_regex = r"(?im)^(\bHPI\b|\bHistory of Present Illness\b|\bInterval History\b)"
blacklist_regex = r"(?im)(\b(Past |Prior )?((Social|Surgical|Family|Medical|Psychiatric|Seizure|Disease|Epilepsy) )?History\b|\bSemiology\b|\bLab|\bExam|\bDiagnostic|\bImpression|\bPlan\b|\bPE\b|\bRisk Factor|\bMedications)"

#get notes from epilepsy specialists
p_notes = p_notes.loc[p_notes['note_author'].str.lower().isin(whitelist_authors)]

#ignore attending attestations
attendings = p_notes['note_text'].apply(lambda x: SequenceMatcher(None, x[:200], addendum_string).ratio() > 0.75)
p_notes = p_notes[~attendings].reset_index(drop=True)

#convert dates to datetimes
p_notes['visit_date'] = p_notes['visit_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

#some parameters to process the notes
splitter = "  "
max_length = 3*512 - 30
#containers to store paragraphs
paragraphs_no_truncation = []
paragraphs_after_truncation = []

#iterate through each note
for v_idx, v_row in p_notes.iterrows():
    #progress counter
    if v_idx % 5000 == 0:
        print(v_idx)
        
    #get paragraphs without truncation
    paragraph_no_trunc = get_paragraph(whitelist_regex, 
                              blacklist_regex, 
                              v_row['note_text'], 
                              v_row['note_author'], 
                              "PAT_ID", 
                              str(v_row['visit_date']), 
                              splitter, 
                              len(v_row['note_text']))
    
    #get paragraphs with truncation
    paragraph_truncated = get_paragraph(whitelist_regex, 
                              blacklist_regex, 
                              v_row['note_text'], 
                              v_row['note_author'], 
                              "PAT_ID", 
                              str(v_row['visit_date']), 
                              splitter, 
                              max_length)
    
    #if the paragraphs are not empty, add them to our containers
    if paragraph_no_trunc != None:
        paragraphs_no_truncation.append(paragraph_no_trunc)
    if paragraph_truncated != None:
        paragraphs_after_truncation.append(paragraph_truncated)
        
#load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

#for each untruncated document, calculate the tokenized length
no_trunc_tokenized_lengths = []
for document in paragraphs_no_truncation:
    for key in document.keys():
        #ignore filename and note_author keys
        if (key == 'filename') or (key == 'note_author'):
            continue
        
        #get the length of the tokenized passage
        no_trunc_tokenized_lengths.append(len(tokenizer(document[key])['input_ids']))
        
#for each truncated document, calculate the tokenized length with the additional appendage of note date
trunc_tokenized_lengths = []
for document in paragraphs_after_truncation:
    
    doc_date = document['filename'].split("_")[3]
    doc_text_append = "This note was written on " + doc_date + ". "
    
    for key in document.keys():
        #ignore filename and note_author keys
        if (key == 'filename') or (key == 'note_author'):
            continue
        doc_text = doc_text_append + document[key]
        #get the length of the tokenized passage
        trunc_tokenized_lengths.append(len(tokenizer(doc_text)['input_ids']))
        
#how many truncated documents exceeded 512 tokens?
print(np.sum(np.array(trunc_tokenized_lengths) > 512))

#plot the tokenized length distributions
fig = plt.figure(figsize=(12,5), dpi=600)
bins = np.linspace(0, max(no_trunc_tokenized_lengths), 500)
plt.hist(no_trunc_tokenized_lengths, bins=bins, alpha = 0.5)
plt.hist(trunc_tokenized_lengths, bins=bins, alpha = 0.5)
plt.vlines(512, 0, 1000, colors='r', linestyles='dashed')
plt.xlabel('Number of Tokens')
plt.ylabel('Number of Paragraphs')
plt.xlim([0, 1500])
plt.ylim([0, 525])
plt.legend(['512 Tokens', 'Untruncated Paragraphs', 'Truncated Paragraphs'])
plt.title("Distribution of Tokens per Paragraph")
fig.savefig('Supplementary_Figure_1.png', bbox_inches='tight')
fig.savefig('Supplementary_Figure_1.pdf', bbox_inches='tight')