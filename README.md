# Extracting-Seizure-Frequency-From-Epilepsy-Clinic-Notes
Code for the paper: Extracting Seizure Frequency From Epilepsy Clinic Notes

# Requirements
1. Python 3.9
2. Huggingface Transformers [Editable Install](https://huggingface.co/transformers/installation.html#editable-install)
3. numpy
4. pandas
5. scikit-learn
6. matplotlib
7. seaborn
8. fuzzywuzzy

# Repository Contents
1. annotation_paragraph_extractor.py - extracts History of Present Illness and/or Interval History sections from our dataset of progress notes to create the annotation paragraphs for the annotation groups.
2. annotation_utils.py - helper functions for collecting and analyzing the annotations
3. combine_INCEpTION_curation_with_annotation.py - collects the adjudicated annotations into the same folders as the human annotations for a given annotation group
4. combine_INCEpTION_curation.py - combines the curated ground truths for all groups into a single folder
5. calculate_INCEpTION_curation_stats.py - calculates some statistics/measures for the ground truth dataset
6. calculate_INCEpTION_project_agreement.py - calculates the intra-group pariwise annotation agreement for a given annotation group
7. generate_training_testing_datasets.py - converts the ground_truth annotation dataset into a squad and boolq formatted dataset, with 70/30 train/test split.
8. generate_annotator_QA_predictions.py - uses human annotations to generate human "predictions", which are then used to calculate human baselines; 
9. combine_notes_for_MLM.py - combines the remaining progress notes not used in annotation into a large text dataset for masked language modelling.
10. boolq_to_transformers_dataset.py - converts a boolq-formatted dataset into the transformers dataset format
11. squad_to_transformers_dataset.py - converts a SQuAD-formatted dataset into the transformers dataset format
12. run_boolqa.py - finetunes or evaluates a transformer model for text classification using a boolq-like dataset. 
13. split_EQA_predictions_by_question.py - splits the predictions of the text extraction questions (Q2 and Q3) into separate files to allow for a question-specific analysis. 
14. boolq_confusion_matrix.py - calculates the confusion matrix for a text classification model
15. evaluate-v2.0_custom.py - a modified version of the official SQuAD v2.0 evaluation script. It skips missing predictions, which occurred when calculating human baselines, as each human annotator only encountered a fraction of the total dataset. 
16. merge_predictions_and_ground_truths.py - merges model predictions with the ground truth datasets into a single json file, allowing for side-by-side comparison between ground truth answers with model predictions for each question and paragraph.
17. plot_results.py - plots the performance metrics of the model
18. calc_stats_for_results.py - performs 2-sided Mann-Whitney U tests on the performance metrics to determine statistical significance.
19. evaluate_tokenized_lengths.py - tokenizes the extracted paragraphs using the BERT tokenizer and measures the distribution of lengths before and after truncation.

# Working Directory Organization
We outline our working directory structure here. For PHI privacy concerns, only the code can be included. 
```
<All code files>
Annotation_Project_Groups
|->Group_1
|->Group_2
|->Group_3
|->Group_4
|->Group_5
|->all_documents_for_annotation
Progress_notes.csv
Group_1_annotations
Group_2_annotations
Group_3_annotations
Group_4_annotations
Group_5_annotations
combined_curation
QA_Datasets
human_baseline_predictions
```
## Directory Descriptions
1. `<All code files>`: the code included in this repository
2. Annotation_Project_Groups: the directory containing the documents (paragraphs) for each annotation group. Documents are stored in their respective Group_X subdirectory, and in all_documents_for_annotation, which contains all documents together.
3. Progress_notes.csv: the raw progress notes, given as a csv table.
4. Group_X_annotations: the annotated documents from group X (see section Annotation Formatting below)
5. combined_curation: the combined set of ground-truth annotations and documents
6. QA_Datasets: contains the training and testing data created from the ground-truth annotations
7. human_baseline_predictions: contains the annotations used to create human baselines in the form of BoolQ and SQuAD predictions. 

## Annotation Formatting
We export all annotations from INCEpTION using the WebAnno TSV c3.2 format. 
Exports have this general file structure:
```
Group_1_annotations
|->annotation
|--->annotation_document_1
|----->annotator_1.tsv
|----->annotator_2.tsv
|----->annotator_3.tsv
|--->(other annotation documents)
|--->annotation_document_200
|----->annotator_1.tsv
|----->annotator_2.tsv
|----->annotator_3.tsv
|->curation
|--->annotation_document_1
|----->CURATION_USER.tsv
|--->(other annotation documents)
|--->annotation_document_200
|----->CURATION_USER.tsv
|->(other subdirectories)
```
annotator_X.tsv files contain annotations for a given annotator on the specified document. Table headers are:
1. Sen-Tok (the sentence and token number of a token)
2. Beg-End (character indices of the beginning and end of the token)
3. Token (the token itself)
4. Remaining columns are devoted to layer names and contain where each annotation occurred. 

# Reproducing our Experiment
We acknowledge that our work cannot be reproduced without the data itself. However, here is the general order of steps.
We used Python 3.8 and 3.9. 
  
## Annotations Generation and Analysis
1. `python annotation_paragraph_extractor.py` (divides 1000 paragraphs into five annotation groups. We manually also collected these documents into Annotation_Project_Groups/all_documents_for_annotation)
2. Perform annotations on INCEpTION and export them into the working directory with names Group_x_annotations, where x is the annotation group
3. `python combine_INCEpTION_curation_with_annotation.py` (collects human annotations with ground truth annotation for easier comparisons. Repeat for each annotation group)
4. `python combine_INCEpTION_curation.py` (collects all ground truth annotations into the same directory)
5. `python calculate_INCEpTION_project_agreement.py` (calculates intragroup iter-rater reliability. Repeat for each annotation group)
6. `python calculate_INCEpTION_curation_stats.py` (calculates statistics for the ground-truth annotations)
7. `python generate_training_testing_datasets.py` (converts ground-truth annotations into SQuAD and BoolQ formatted training and testing datasets)
8. `python combine_notes_for_MLM.py` (creates text datasets for Masked Language Modelling)
9. `python generate_human_baselines.py` (creates human baselines in the form of SQuAD and BoolQ formatted predictions)

## Model Finetuning Pipeline
1. We moved the annotated training and testing datasets onto a HIPAA-protected computing service at the Penn Medicine Digital Academic Research Transformation. We utilize an Nvidia P100 16GB GPU for finetuning and evaluation purposes. 
2. We convert the BoolQ3L dataset, and our annotated training and testing sets from the SQuAD and BoolQ format to the Transformers Dataset format. For example
  ```
  python squad_to_transformers_dataset.py \
  --file_path PATH_TO_SQuAD_FORMAT_DATASET \
  --save_path PATH_TO_TRANSFORMERS_TEXT_EXTRACTION_DATASET
  
  python boolq_to_transformers_dataset.py \
  --file_path PATH_TO_BOOLQ_FORMAT_DATASET \
  --save_path PATH_TO_TRANSFORMERS_CLASSIFICATION_DATASET
  ```
3. Finetune with masked language modelling. We use Huggingface's [`run_mlm.py`](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling). For example:
```
python transformers/examples/pytorch/language-modeling/run_mlm.py \
--model_name_or_path bert-base-cased \
--train_file note_text_for_mlm_1.txt \
--do_train \
--seed 17 \
--cache_dir CACHE \
--overwrite_cache \
--output_dir BB_MLM_17/part_1 \
--max_seq_length 512 \
--save_steps 15000
```
4. Finetune with SQuADv2. We use Huggingface's [`run_qa.py`](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering). For example: 
```
python transformers/examples/pytorch/question-answering/run_qa.py \
--model_name_or_path BB_MLM_17/ \
--dataset_name squad_v2 \
--cache_dir CACHE \
--overwrite_cache \
--do_train \
--do_eval \
--seed 17 \
--version_2_with_negative \
--output_dir BB_MLM_SQ_17 \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 4 \
--max_seq_length 512 \
--doc_stride 128 \
--save_steps 25000
```
5. Finetune with BoolQ3L. For example:
```
python run_boolqa.py \
--model_name_or_path BB_MLM_17/ \
--train_file boolq3L_train.json \
--validation_file boolq3L_test.json \
--do_train \
--do_eval \
--seed 17 \
--cache_dir CACHE \
--output_dir BB_MLM_B3_17 \
--max_seq_length 512 \
--save_steps 5000 \
--num_labels 3
```
6. Finetune with the annotated training set and evaluate on the testing set. Again, we use Huggingface's `run_qa.py`, and our `run_boolqa.py` as above, but with the appropriate `model_name_or_path`, `train_file`, `validation_file`, and `output_dir`. 
7. We conduct ablation studies by eliminating a finetuning step accordingly.
8. We conduct training size analysis using the above steps, with the appropriate `train_file` that was generated when running `generate_training_testing_datasets.py` for the final finetuning operation. 

## Performance Evaluation
1. Split the text extraction predictions by question to allow for a question-based analysis. For example, 
```
python split_EQA_predictions_by_question.py \
--ground_truth_path PATH_TO_ANNOTATED_TEST_FILE \
--model_pred_path PATH_TO_PREDICTIONS  \
--output_path PATH_TO_OUTPUT_DIRECTORY
```
2. Use `evaluate-2.0_custom.py`, `boolq_confusion_matrix.py`, or the evaluation outputs from `run_boolq.py` and `run_qa.py` to calculate performance metrics of the human baselines and the model. For example,
```
python evaluate-2.0_custom \
text_extraction_test.json \
PQF_predictions.json

python boolq_confusion_matrix \
classification_annotator_0_predictions.tsv
```
4. `python plot_results.py` (plots the results. Our performance metrics are hard-coded into the file)
5. `calc_stats_for_results.py` (calculates statistical tests. Our performance metrics are hard-coded into the file). 
