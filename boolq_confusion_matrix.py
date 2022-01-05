import json
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

#Input arguments
parser = argparse.ArgumentParser(description='Calculates the confusion matrix between a boolq-like dataset and model predictions')
parser.add_argument("file_path", help='The path to the predictions (.tsv)')
args = parser.parse_args()


#read the predictions file
data = pd.read_csv(args.file_path, sep='\t', header=0)
truth_labels = data['True_Label'].tolist()
predictions_str = ["".join(ch for ch in pred if ch not in '[]') for pred in data['Predictions']]
predictions_split = [[float(confidence) for confidence in pred.split()] for pred in predictions_str]
predictions_labels = [np.argmax(pred) for pred in predictions_split]

#print labels
print("Ground truth labels:")
print(truth_labels)
print("=========")
print("Predictions:")
print(predictions_labels)
print("==========")
print("Label differences")
print([truth_labels[i] - predictions_labels[i] for i in range(len(truth_labels))])

#calculate the confusion matrix
conf_mat = confusion_matrix(truth_labels, predictions_labels)
print("Accuracy: " + str(accuracy_score(truth_labels, predictions_labels)))
print(conf_mat)
