import pandas as pd
import numpy as np
import re
import copy
import random
import os

#Initial file output
initial_output = "note_text_for_mlm"

#how many notes per file?
npf = 40000

#What directory of file names should be excluded (i.e. the complete list of notes that have been/will be annotated)
exclusion_dir = 'Annotation_Project_Groups/all_documents_for_annotation'
excluded_prototypes = ["".join(filename.split('.')[0].split('_')[:-1]) for filename in os.listdir(exclusion_dir)]

#read the notes
c_notes = pd.read_csv("Progress_Notes.csv")[['pat_id', 'visit_date', 'note_author', 'note_text']]

total_notes_used = 0 #how many notes did we use for MLM?
splitter = "  " #split between new lines in the note text
file_idx = 0 #mlm dataset file index if we're splitting the dataset into parts

#for each note in our dataset
for c_index, c_row in c_notes.iterrows():
    
    #counter
    if c_index % 5000 == 0:
        print("Starting " + str(c_index))
    
    #what is the name info of this file?
    prototype_name = str(c_row['pat_id'])+str(c_row['note_author'])+str(c_row['visit_date'][:10])

    #check if this file was annotated
    if prototype_name in excluded_prototypes:
        continue
    
    #get the document's text
    extracted_paragraphs = c_row['note_text'].strip().split(splitter)
    doc_text = "".join([line.strip() + "\n" for line in extracted_paragraphs if line != ""])
    
    #have we filled the current file with notes?
    if total_notes_used%npf==0:
        file_idx+=1
    
    #save the note into the file
    with open(initial_output+"_"+str(file_idx)+".txt", 'a') as f:
        f.write(doc_text)
    
    total_notes_used += 1
    
print(total_notes_used)
