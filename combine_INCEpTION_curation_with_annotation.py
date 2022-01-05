import pandas as pd
import numpy as np
import csv
import os
from shutil import copyfile

#what is the exported project's local directory?
proj_dir = 'Group_1_annotations'

#Check if this project is correctly set up. 
proj_subdirs = os.listdir(proj_dir)
if "curation" not in proj_subdirs or "annotation" not in proj_subdirs:
    raise Exception

#create a new subdirectory called combined 
if "combined" not in proj_dir:
    os.mkdir(proj_dir+"/combined")

#populate the combined dirctory with subdirectories of annotation and curation documents
for curated_dir in os.listdir(proj_dir+"/curation"):
    os.mkdir(proj_dir+"/combined/"+curated_dir)
    
    #copy the curated annotations over
    copyfile(proj_dir+"/curation/"+curated_dir+"/CURATION_USER.tsv", proj_dir+"/combined/"+curated_dir+"/CURATION_USER.tsv")
    
    #for each annotation file in the annotation subdirectory
    for anno_file in os.listdir(proj_dir+"/annotation/"+curated_dir):
        if 'ipynb_checkpoints' in anno_file:
            continue
        copyfile(proj_dir+"/annotation/"+curated_dir+"/"+anno_file, proj_dir+"/combined/"+curated_dir+"/"+anno_file)
