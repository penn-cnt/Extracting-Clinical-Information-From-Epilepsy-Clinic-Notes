"""
Trains a model on a bool-qa like dataset.
Follows code and discussion from: https://discuss.huggingface.co/t/question-answering-bot-yes-no-answers/4496
and from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
"""

import json
import argparse
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed


#Input arguments
parser = argparse.ArgumentParser(description='Trains a model on a bool-qa like dataset')
parser.add_argument("--model_name_or_path", help='The name of the dataset in huggingface hubs, or the path to its folder',\
                    required=True)
parser.add_argument("--train_file", help="The path to the training set (.json)",\
                    required=False)
parser.add_argument('--validation_file', help="The path to the validation set (.json)", \
                    required=False)
parser.add_argument('--cache_dir', help="The cache dir", required=True)
parser.add_argument('--save_steps', help='The number of steps before a checkpoint is saved', required=True, type=int)
parser.add_argument('--max_seq_length', help="The maximum tokenized length of the model", required=True, type=int)
parser.add_argument('--output_dir', help='Where to save the trained model', required=True)
parser.add_argument('--num_labels', help='The number of classification labels (UP TO 3)',required=True, type=int)
parser.add_argument('--do_train', help='Run training?', required=False, action='store_true')
parser.add_argument('--do_eval', help="Run evaluation?", required=False, action='store_true')
parser.add_argument('--seed', help='What seed do you want?', required=False, default=42, type=int)
args = parser.parse_args()

#set the seed
set_seed(args.seed)

#tokenizer for the dataset
def tokenize_function(examples):
    return tokenizer(examples['question'], examples['passage'], 
                         max_length=args.max_seq_length, 
                         stride=128, 
                         truncation="only_second",
                         padding='max_length')

#performance metrics
metric_acc = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

#prepare the raw dataset
if args.do_train and args.do_eval:
    raw_dataset = load_dataset('json', 
                               data_files={'train':args.train_file, 'validation':args.validation_file},
                               cache_dir=args.cache_dir)
elif args.do_train:
    raw_dataset = load_dataset('json', 
                               data_files={'train':args.train_file},
                               cache_dir=args.cache_dir)
elif args.do_eval:
    raw_dataset = load_dataset('json', 
                               data_files={'validation':args.validation_file},
                               cache_dir=args.cache_dir)
else:
    raise ValueError("No training or validation specified")


#load in a tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, num_labels=args.num_labels)

#tokenize the dataset
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, load_from_cache_file = False)

#training arguments
train_args = TrainingArguments(output_dir=args.output_dir,
                               per_device_train_batch_size=4,
                               learning_rate=3e-5,
                               num_train_epochs=4,
                               save_steps=args.save_steps
                               )

#prepare the trainer
if args.do_train:
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset['train'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
elif args.do_eval: 
    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

if args.do_train:
    #train the model
    training_results = trainer.train()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics('train', training_results.metrics)
    trainer.save_metrics('train', training_results.metrics)

if args.do_eval:
    #make predictions and evaluate on the validation set
    validation_results = trainer.predict(tokenized_dataset['validation'])
    trainer.log_metrics("eval", validation_results.metrics)
    trainer.save_metrics("eval", validation_results.metrics)
    #save predictions to file
    with open(args.output_dir+"/eval_predictions.tsv", 'w') as f:
        f.write('Predictions\tTrue_Label')
        f.write('\n')
        for idx in range(len(validation_results.predictions)):
            f.write(str(validation_results.predictions[idx]))
            f.write('\t')
            f.write(str(validation_results.label_ids[idx]))
            f.write('\n')
