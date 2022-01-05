import pandas as pd
import numpy as np
import csv
import os
import string
from functools import reduce
from sklearn.metrics import cohen_kappa_score

def get_doc_by_name(doc_list, doc_name):
    """Finds a Document in a list of Documents by the Document's name"""
    for doc in doc_list:
        if doc.name == doc_name:
            return doc
    raise ValueError("Document with specified name not found")

def check_span_overlap(start_idx_1, end_idx_1, start_idx_2, end_idx_2):
    """check if two spans, defined by starting and ending indicies, overlap with each other"""
    return (start_idx_1 in range(start_idx_2, end_idx_2+1)) or (start_idx_2 in range(start_idx_1, end_idx_1+1))

def series_to_paragraph(series):
    """Turns a pd.series into a paragraph of text. Each element of the series is added to the previous following a " " (space)"""
    return "".join([" " + str(element) if element not in string.punctuation else element for element in series.tolist()]).strip()

def GetHeaders(tsv_path):
    """gets the headers from an INCEpTION tsv export"""
    headers = ['Sen-Tok','Beg-End', 'Token']
    with open(tsv_path) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            if line:
                if '#T_SP' in line[0]:
                    headers.append(line[0][line[0].find('|')+1:])
                elif '#T_CH' in line[0]:
                    first = line[0].find('|') +1
                    second = line[0].find('|', first+1)+1
                    headers.append(line[0][first:second-1])
                    headers.append(line[0][second:])
    return headers

def collect_annotations(annotations_dataframe, Doc, all_annotators):
    """Collect annotations for each annotator."""  
    
    #for each word in this document
    for index, row in annotations_dataframe.iterrows():
        
        #for each column (annotation layer) in this document
        for layer in annotations_dataframe.columns:
            
            #if there is no annotation layer at the current word, skip this layer
            if row[layer] == "_":
                continue
            
            #otherwise, get the type of layer and the annotator index
            layer_info = layer.split("_")
            
            #if multiple annotations exist at this word for this annotation layer and annotator, split it up and process each annotation separately
            if "|" in row[layer]:
                subsets = row[layer].split("|")
                
                #for each sub-annotation in the multiple annotation block
                for subset in subsets:
                    #create an annotation and check if it already exists for this annotator
                    current_anno = Annotation(all_annotators[int(layer_info[1])], subset, Doc, layer_info[0], index, index)            
                    anno_exists = all_annotators[int(layer_info[1])].check_annotation_exists(current_anno)

                    #if it already exists, increment the ending index
                    if anno_exists >= 0:
                        all_annotators[int(layer_info[1])].annotations[anno_exists].set_end_idx(index)
                    else:
                    #otherwise, add this new annotation to the Annotator and the Document
                        all_annotators[int(layer_info[1])].add_annotation(current_anno)
                        Doc.add_annotation(current_anno)
                        if all_annotators[int(layer_info[1])] not in Doc.annotators:
                            Doc.add_annotator(all_annotators[int(layer_info[1])])
                        
            else:
                
                #see documentation above
                current_anno = Annotation(all_annotators[int(layer_info[1])], row[layer], Doc, layer_info[0], index, index)            
                anno_exists = all_annotators[int(layer_info[1])].check_annotation_exists(current_anno)

                if anno_exists >= 0:
                    all_annotators[int(layer_info[1])].annotations[anno_exists].set_end_idx(index)
                else:
                    all_annotators[int(layer_info[1])].add_annotation(current_anno)
                    Doc.add_annotation(current_anno)
                    if all_annotators[int(layer_info[1])] not in Doc.annotators:
                        Doc.add_annotator(all_annotators[int(layer_info[1])])

#Document class
#A document has: 
    #its name
    #its text
    #its list of Annotations
    #its list of Annotators
    #the list of ground_truths
class Document():
    def __init__(self, name):
        self.name = name
        self.text = ""
        self.annotations = []
        self.annotators = []
        self.ground_truths = []
        
    def add_annotation(self, Annotation):
        self.annotations.append(Annotation)
        
    def add_ground_truth(self, ground_truth):
        self.ground_truths.append(ground_truth)
        
    def add_annotator(self, Annotator):
        self.annotators.append(Annotator)
        
    def set_text(self, text):
        self.text = text
        
    def get_raw_text(self):
        """Turns the pd.series self.text into a paragraph of text. Each element of the series is added to the previous following a " " (space)"""
        return series_to_paragraph(self.text['Token'])

    def info(self):
        print("Document: " + str(self))
        print("Name: " + self.name)
        print("Text: " + self.get_raw_text())
        print("Annotators: " + str(self.annotators))
        print("Annotations: " + str(self.annotations))
        
#Define an annotator class
#Annotators have
    #A name
    #A list of Annotation objects
    #An annotator index
    #A list of documents annotated
class Annotator:
   
    #constructor
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx
        self.annotations = []
        self.documents = []
    
    def check_annotation_exists(self, new_Annotation):
        """Check if an annotation exists for this annotator"""
        idx = 0
        for annotation in self.annotations:
            if new_Annotation.Annotator == self and \
            annotation.value == new_Annotation.value and \
            annotation.Document == new_Annotation.Document:
                return idx
            idx += 1
        return -999
        
    def add_annotation(self, Annotation):
        self.annotations.append(Annotation)
        
    def add_document(self, Document):
        self.documents.append(Document)
        
    def info(self):
        print("Printing: " + str(self))
        print("Name: " + str(self.name))
        print("Number of Annotations: " + str(len(self.annotations)))
        print("Documents Annotated: " + str(self.documents))
        print("==============================================")
        
#Define an annotation class
#Annotations have
    #An annotator
    #A value
    #A document object
    #A starting index
    #An ending index
    #Overlap with other annotations 
    #An annotation layer (hasSz, etc...)
class Annotation:
   
    #constructor
    def __init__(self, Annotator, value, Document, layer, start_idx, end_idx):
        self.Annotator = Annotator
        self.value = value
        self.Document = Document
        self.layer = layer
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.overlaps = []
    
    def set_end_idx(self, end_idx):
        self.end_idx = end_idx
    def set_start_idx(self, start_idx):
        self.start_idx = start_idx
        
    def check_overlap(self, other_Annotation):
        """check if two annotations overlap with each other"""
        return self.Document == other_Annotation.Document \
            and self.layer == other_Annotation.layer \
            and check_span_overlap(self.start_idx, self.end_idx, other_Annotation.start_idx, other_Annotation.end_idx) 
    
    def add_overlap(self, other_Annotation):
        self.overlaps.append(other_Annotation)
    
    def get_raw_value(self):
        """returns the raw value of the annotation (Yes, No, GTC.....)"""
        return "".join([ch for ch in self.value if ch.isalpha()])
    
    def get_text(self):
        """returns the corresponding text of the annotation using the start and ending indices"""
        return series_to_paragraph(self.Document.text['Token'].iloc[self.start_idx:self.end_idx+1])        
    
    def info(self):
        print("Printing: " + str(self))
        print("Annotator: " + str(self.Annotator.name))
        print("Value: " + str(self.value))
        print("Document: " + str(self.Document.name))
        print("Layer: " + str(self.layer))
        print("Starting Index: " + str(self.start_idx))
        print("Ending Index: " + str(self.end_idx))
        print("Overlaps: " + str(self.overlaps))
        print("==============================================")


#agreement class.
#An agreement has:
    #Two different annotators (unordered set)
    #Classification agreement metrics
    #Text Extraction agreement metrics
    #Seizure Type Extraction agreement metrics
#Abbreviations: 
#SSSA: Same Span Same Annotation - the highlighted spans and annotations were identical
#SSDA: Same Span Different Annotation - the highlighted spans were identical, but the annotations were different
#OSSA: Overlapping Span Same Annotation - the highlighted spans overlapped each other, and the annotations were identical
#OSDA: Overlapping Span Different Annotation - the highlighted spans overlapped each other, and the annotations were different
#DSSA: Different Span Same Annotation - the highlighted spans were completely different from each other, but the annotations were the same (classification only)
#DSDA: Different Span Different Annotation - the highlighted spans were completely different from each other, and the annotations were different (classification only) 
#DS: Different Span - the highlighted spans were completely different from each other, and no annotation relationship can be determined
class Agreement():
    def __init__(self, Annotator_1, Annotator_2):
        if Annotator_1 == Annotator_2:
            raise TypeError("Error: Annotators must be different") 
        self.annotators = {Annotator_1, Annotator_2}
        self.least_annotated_annotator = Annotator_1 \
            if len(Annotator_1.documents) <= len (Annotator_2.documents) \
            else Annotator_2
        self.most_annotated_annotator = Annotator_1 \
            if len(Annotator_1.documents) > len (Annotator_2.documents) \
            else Annotator_2
        self.hasSz_values = []
        self.szFreq_values = []
        self.szType_values = []
        self.hasSz_metrics = {'total_num_annotations':0}
        self.szFreq_metrics = {'total_num_annotations':0}
        self.szType_metrics = {'total_num_annotations':0}
        self.hasSz_agreement = {'SSSA':0, \
                               'SSDA':0, \
                               'OSSA':0, \
                               'OSDA':0, \
                               'DSSA':0, \
                               'DSDA':0, \
                               'DS':0}
        self.szFreq_agreement = {'SSSA':0, \
                               'SSDA':0, \
                               'OSSA':0, \
                               'OSDA':0, \
                               'DSSA':0, \
                               'DSDA':0, \
                               'DS':0}
        self.szType_agreement = {'SSSA':0, \
                               'SSDA':0, \
                               'OSSA':0, \
                               'OSDA':0, \
                               'DSSA':0, \
                               'DSDA':0, \
                               'DS':0}
        self.hasSz_pairs = []
        self.szFreq_pairs = []
        self.szType_pairs = []
    
    def info(self):
        print("Printing: " + str(self))
        print()
        print("Annotators: " + str(self.annotators))
        print()
        print("HasSz Metrics: " + str(self.hasSz_metrics))
        print()
        print("SzFreq Metrics: " + str(self.szFreq_metrics))
        print()
        print("SzType Metrics: " + str(self.szType_metrics))
        print()
        print("HasSz Agreement: "+ str(self.hasSz_agreement))
        print()
        print("SzFreq Agreement: "+ str(self.szFreq_agreement))
        print()
        print("SzType Agreement: "+ str(self.szType_agreement))
    
    def __calc_DSXX_agreement(self):
        """Calculate DSXX agreement for hasSz annotations"""
        
        #for each document that the least-annotated annotator has done,
        for doc in self.least_annotated_annotator.documents:
            
            #check if the other annotator has also annotated this document
            if doc not in self.most_annotated_annotator.documents:
                continue
                
            #for each annotation that the least-annotated annotator has done,
            for annotation_1 in self.least_annotated_annotator.annotations:
                
                #skip if wrong document, or if the layer isn't hasSz
                if annotation_1.Document != doc or annotation_1.layer != 'HasSeizures':
                    continue
                    
                #for each annotation that the most-anotated annotator has done
                for annotation_2 in self.most_annotated_annotator.annotations:

                    #skip if wrong document or wrong layer
                    if annotation_2.Document != doc or annotation_2.layer != 'HasSeizures':
                        continue
                        
                    #check if either of the two annotations have overlaps, and if that overlap was with these annotators
                    correct_annotator_overlap = False
                    for overlap in annotation_1.overlaps:
                        if overlap.Annotator == self.most_annotated_annotator:
                            correct_annotator_overlap = True
                            break
                    for overlap in annotation_2.overlaps:
                        if overlap.Annotator == self.least_annotated_annotator:
                            correct_annotator_overlap = True
                            break
                    if correct_annotator_overlap:
                        continue
                    
                    #do the annotations match?
                    if annotation_1.get_raw_value() == annotation_2.get_raw_value():
                        self.hasSz_agreement['DSSA'] += 1
                    else:
                        self.hasSz_agreement['DSDA'] += 1
                        
                    self.hasSz_pairs.append({annotation_1.Annotator.name:annotation_1.get_raw_value(),
                                                                annotation_2.Annotator.name:annotation_2.get_raw_value()})
    def calc_simple_agreement(self):
        """Calculate SSXX, OSXX, and DS agreement for all annotations"""
        #for each annotator
        for annotator in self.annotators:
            #for each annotation
            for annotation in annotator.annotations:
                
                #skip if this annotation is from a document that only one annotator has annotatoed
                if annotation.Document not in self.least_annotated_annotator.documents or \
                annotation.Document not in self.most_annotated_annotator.documents:
                    continue
                
                #get the raw value of the annotation
                annotation_value = annotation.get_raw_value()
                
                #add the annotation value to the lists of annotated values
                if annotation.layer == "HasSeizures" and annotation_value not in self.hasSz_values:
                    self.hasSz_values.append(annotation_value)
                elif annotation.layer == "SeizureFrequency" and annotation_value not in self.szFreq_values:
                    self.szFreq_values.append(annotation_value)
                elif annotation.layer == "TypeofSeizure" and annotation_value not in self.szType_values:
                    self.szType_values.append(annotation_value)
                
                #make sure the annotation's value is a keyword in the dictionaries
                if annotation.layer == "HasSeizures" and annotation_value+"_"+annotator.name not in self.hasSz_metrics:
                    self.hasSz_metrics[annotation_value+"_"+annotator.name] = 0
                elif annotation.layer == "SeizureFrequency" and annotation_value+"_"+annotator.name not in self.szFreq_metrics:
                    self.szFreq_metrics[annotation_value+"_"+annotator.name] = 0
                elif annotation.layer == "TypeofSeizure" and annotation_value+"_"+annotator.name not in self.szType_metrics:
                    self.szType_metrics[annotation_value+"_"+annotator.name] = 0
                
                #if this annotation does not overlap any other annotation, it must be a DS
                if not annotation.overlaps:
                    if annotation.layer == "HasSeizures":
                        self.hasSz_agreement['DS'] += 1
                        self.hasSz_metrics[annotation_value+"_"+annotator.name] += 1
                        self.hasSz_metrics["total_num_annotations"] += 1
                    elif annotation.layer == "SeizureFrequency":
                        self.szFreq_agreement['DS'] += 1
                        self.szFreq_metrics[annotation_value+"_"+annotator.name] += 1
                        self.szFreq_metrics["total_num_annotations"] += 1
                    elif annotation.layer == "TypeofSeizure":
                        self.szType_agreement['DS'] += 1
                        self.szType_metrics[annotation_value+"_"+annotator.name] += 1
                        self.szType_metrics["total_num_annotations"] += 1
                    else:
                        print("ERROR: Unknown annotation layer")
                        annotation.info()
                #otherwise, it overlaps with atleast an annotation
                else:
                    #for each overlapping annotation, check if it's from the other annotator
                    #if it is, then it must be one of the SS or OS annotations
                    #if no overlapping annotation is from the other annotator, then it must be a DS
                    correct_annotator_overlap = False
                    for overlap in annotation.overlaps:
                        if overlap.Annotator in self.annotators:
                            correct_annotator_overlap = True
                            
                            #if the spans are the same, it must be SS
                            if overlap.start_idx == annotation.start_idx and overlap.end_idx == annotation.end_idx:
                                #if the values are the same, it must be SSSA
                                if overlap.get_raw_value() == annotation.get_raw_value():
                                    if annotation.layer == "HasSeizures":
                                        self.hasSz_agreement['SSSA'] += 1
                                        self.hasSz_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.hasSz_metrics["total_num_annotations"] += 1
                                        self.hasSz_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "SeizureFrequency":
                                        self.szFreq_agreement['SSSA'] += 1
                                        self.szFreq_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szFreq_metrics["total_num_annotations"] += 1
                                        self.szFreq_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "TypeofSeizure":
                                        self.szType_agreement['SSSA'] += 1
                                        self.szType_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szType_metrics["total_num_annotations"] += 1
                                        self.szType_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    else:
                                        print("ERROR: Unknown annotation layer")
                                        annotation.info()
                                #otherwise, it's SSDA
                                else:
                                    if annotation.layer == "HasSeizures":
                                        self.hasSz_agreement['SSDA'] += 1
                                        self.hasSz_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.hasSz_metrics["total_num_annotations"] += 1
                                        self.hasSz_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "SeizureFrequency":
                                        self.szFreq_agreement['SSDA'] += 1
                                        self.szFreq_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szFreq_metrics["total_num_annotations"] += 1
                                        self.szFreq_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "TypeofSeizure":
                                        self.szType_agreement['SSDA'] += 1
                                        self.szType_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szType_metrics["total_num_annotations"] += 1
                                        self.szType_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    else:
                                        print("ERROR: Unknown annotation layer")
                                        annotation.info()
                            #otherwise, it's an OS
                            else:
                                #if the values are the same, it must be OSSA
                                if overlap.get_raw_value() == annotation.get_raw_value():
                                    if annotation.layer == "HasSeizures":
                                        self.hasSz_agreement['OSSA'] += 1
                                        self.hasSz_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.hasSz_metrics["total_num_annotations"] += 1
                                        self.hasSz_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "SeizureFrequency":
                                        self.szFreq_agreement['OSSA'] += 1
                                        self.szFreq_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szFreq_metrics["total_num_annotations"] += 1
                                        self.szFreq_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "TypeofSeizure":
                                        self.szType_agreement['OSSA'] += 1
                                        self.szType_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szType_metrics["total_num_annotations"] += 1
                                        self.szType_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    else:
                                        print("ERROR: Unknown annotation layer")
                                        annotation.info()
                                #otherwise, it's OSDA
                                else:
                                    if annotation.layer == "HasSeizures":
                                        self.hasSz_agreement['OSDA'] += 1
                                        self.hasSz_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.hasSz_metrics["total_num_annotations"] += 1
                                        self.hasSz_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "SeizureFrequency":
                                        self.szFreq_agreement['OSDA'] += 1
                                        self.szFreq_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szFreq_metrics["total_num_annotations"] += 1
                                        self.szFreq_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    elif annotation.layer == "TypeofSeizure":
                                        self.szType_agreement['OSDA'] += 1
                                        self.szType_metrics[annotation_value+"_"+annotator.name] += 1
                                        self.szType_metrics["total_num_annotations"] += 1
                                        self.szType_pairs.append({annotation.Annotator.name:annotation.get_raw_value(),
                                                                overlap.Annotator.name:overlap.get_raw_value()})
                                    else:
                                        print("ERROR: Unknown annotation layer")
                                        annotation.info()
                    
                    #if no overlaps were with the other annotator in this agreement pair, it must be a DS
                    if not correct_annotator_overlap:
                        if annotation.layer == "HasSeizures":
                            self.hasSz_agreement['DS'] += 1
                            self.hasSz_metrics[annotation_value+"_"+annotator.name] += 1
                            self.hasSz_metrics["total_num_annotations"] += 1
                        elif annotation.layer == "SeizureFrequency":
                            self.szFreq_agreement['DS'] += 1
                            self.szFreq_metrics[annotation_value+"_"+annotator.name] += 1
                            self.szFreq_metrics["total_num_annotations"] += 1
                        elif annotation.layer == "TypeofSeizure":
                            self.szType_agreement['DS'] += 1
                            self.szType_metrics[annotation_value+"_"+annotator.name] += 1
                            self.szType_metrics["total_num_annotations"] += 1
                        else:
                            print("ERROR: Unknown annotation layer")
                            annotation.info()
                
        #calculate the DSSA and DSDA agreements
        self.__calc_DSXX_agreement()
        
        #Overlapping span and same span counters will be double counted. divide by two
        self.szType_agreement['SSSA'] /= 2
        self.szType_agreement['SSDA'] /= 2
        self.szType_agreement['OSSA'] /= 2
        self.szType_agreement['OSDA'] /= 2
        
        self.szFreq_agreement['SSSA'] /= 2
        self.szFreq_agreement['SSDA'] /= 2
        self.szFreq_agreement['OSSA'] /= 2
        self.szFreq_agreement['OSDA'] /= 2
        
        self.hasSz_agreement['SSSA'] /= 2
        self.hasSz_agreement['SSDA'] /= 2
        self.hasSz_agreement['OSSA'] /= 2
        self.hasSz_agreement['OSDA'] /= 2
        
    def calc_cohen_kappa(self, layer):
        """Calculate cohen's kappa with specific metrics and agreement
        https://en.wikipedia.org/wiki/Cohen%27s_kappa  
        """
        try:
            least_annotator_list = []
            most_annotator_list = []
            if layer == "HasSeizures":
                for pair in self.hasSz_pairs:
                    least_annotator_list.append(pair[self.least_annotated_annotator.name])
                    most_annotator_list.append(pair[self.most_annotated_annotator.name])
            elif layer == "SeizureFrequency":
                for pair in self.szFreq_pairs:
                    least_annotator_list.append(pair[self.least_annotated_annotator.name])
                    most_annotator_list.append(pair[self.most_annotated_annotator.name])
            elif layer == "TypeofSeizure":
                for pair in self.szType_pairs:
                    least_annotator_list.append(pair[self.least_annotated_annotator.name])
                    most_annotator_list.append(pair[self.most_annotated_annotator.name]) 
            else:
                raise Exception
                
            return cohen_kappa_score(least_annotator_list, most_annotator_list)
                
        except Exception as e:
            print("========================")
            print(e)
            print("========================")

    def calc_average_F1_overlap(self, layer):
        """Calculate the F1 span overlap score
        https://en.wikipedia.org/wiki/F-score
        """
        try:
            cumulative_overlap_score = 0
            total_annotations_considered = 0
            total_overlapped_annotations = 0

            #for each annotator
            for annotator in self.annotators:
                #for each annotation
                for annotation in annotator.annotations:

                    #check if the annotation is of the right layer
                    if annotation.layer != layer:
                        continue

                    #skip if this annotation is from a document that only one annotator has annotated
                    if annotation.Document not in self.least_annotated_annotator.documents or \
                    annotation.Document not in self.most_annotated_annotator.documents:
                        continue

                    #if this annotation has no overlap, accumulate in total_annotations_considered
                    if not annotation.overlaps:
                        total_annotations_considered += 1
                    else:
                        correct_annotator_overlap = False
                        for overlap in annotation.overlaps:
                            #check if the annotation is of the right layer
                            if annotation.layer != layer:
                                continue

                            if overlap.Annotator in self.annotators:
                                correct_annotator_overlap = True
                                break
                        if not correct_annotator_overlap:
                            total_annotations_considered += 1

            #for each annotation done by the annotator with the least annotations
            for annotation in self.least_annotated_annotator.annotations:

                #check if the annotation is of the right layer
                if annotation.layer != layer:
                    continue

                #find correct overlaps and calculate F1 score when they happen
                for overlap in annotation.overlaps:
                    if overlap.Annotator not in self.annotators:
                        continue
                        
                    #find spans to get true positive, and false negative/positive
                    span_1 = set(range(annotation.start_idx, annotation.end_idx + 1))
                    span_2 = set(range(overlap.start_idx, overlap.end_idx+1))
                    tp = span_1 & span_2
                    fp = span_2 - tp
                    fn = span_1 - tp
                    cumulative_overlap_score += len(tp) / (len(tp) + 0.5 * (len(fp) + len(fn)))
                    total_annotations_considered += 1
                    total_overlapped_annotations += 1
            return {"Average overall overlap": cumulative_overlap_score / total_annotations_considered, \
                    "Average paired overlap": cumulative_overlap_score / total_overlapped_annotations}
        except Exception as e:
            print("========================")
            print(e)
            print("========================")

            
