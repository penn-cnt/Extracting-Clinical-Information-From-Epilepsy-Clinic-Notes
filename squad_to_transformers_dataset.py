import json
import argparse

#Input arguments
parser = argparse.ArgumentParser(description='Convert a Squadv2 dataset into the required json format for Transformers datasets')
parser.add_argument("--file_path", help='The path to the dataset (.json)',\
                    required=True)
parser.add_argument("--save_path", help="The path to where to save the formatted datset",\
                    required=True)
args = parser.parse_args()

#open the dataset file
with open(args.file_path) as f:
    squad_normal = json.load(f)['data']

#create an empty list for the transformed dataset.
#each element of this list will be a json object.
dataset = []

for group in squad_normal:
    for paragraph in group['paragraphs']:
        for qa in paragraph['qas']:
            if qa['is_impossible']:
                dataset.append({'answers':{'answer_start':[],
                                           'text':[]},
                                'id':qa['id'],
                                'question':qa['question'],
                                'context':paragraph['context'],
                                'title':group['title']})
            else:
                answers = {'answer_start':[], 'text':[]}
                
                for ans in qa['answers']:
                    answers['answer_start'].append(ans['answer_start'])
                    answers['text'].append(ans['text'])
                    
                dataset.append({'answers':answers,
                                'id':qa['id'],
                                'question':qa['question'],
                                'context':paragraph['context'],
                                'title':group['title']})

output = {'version':"2.0",
          'data':dataset}

with open(args.save_path, 'w') as f:
    json.dump(output, f)
                