import pandas as pd
import numpy as np
import csv
import os
from shutil import copyfile

#What are the curated project directories
curation_dirs = ['Group_1_annotations/curation',
                 'Group_2_annotations/curation',
                 'Group_3_annotations/curation',
                 'Group_4_annotations/curation',
                'Group_5_annotations/curation']

#What is the combined directory?
combined_dir = 'combined_curation'

#attempt to make the directory
try:
    os.mkdir(combined_dir)
except FileExistsError:
    print("Directory already exists")

#copy each curated document over to the combined directory
for curated_dir in curation_dirs:
    curated_annos = os.listdir(curated_dir)
    
    #copy each file over
    for curation in curated_annos:
        os.mkdir(combined_dir+"/"+curation)
        copyfile(curated_dir+"/"+curation+"/CURATION_USER.tsv", combined_dir+"/"+curation+"/CURATION_USER.tsv")
