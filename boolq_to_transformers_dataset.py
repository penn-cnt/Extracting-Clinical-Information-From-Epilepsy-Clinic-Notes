import json
import argparse

#Input arguments
parser = argparse.ArgumentParser(description='Convert a boolq dataset into the required json format for Transformers datasets')
parser.add_argument("--file_path", help='The path to the dataset (.json)',\
                    required=True)
parser.add_argument("--save_path", help="The path to where to save the formatted datset",\
                    required=True)
args = parser.parse_args()

#create a 1-hot-encoding dictionary, assuming True, False, no-answer. Yes and No
#map equally to true and false. 
#if -1, there was an error in the annotation process. map as no-answer
one_hot = {"true":1, "false":0, "no-answer":2, 'yes':1, 'no':0, '-1':2}

#open the datafile and reformat it. 
with open(args.file_path) as f:
  dataset_txt = f.readlines()
  dataset_json = []
  line_cnt = 0
  for datum in dataset_txt:
    datum_json = json.loads(datum)
    datum_json['label'] = one_hot[str(datum_json['answer']).lower()]
    datum_json['idx'] = line_cnt
    del datum_json['answer']
    del datum_json['title']
    dataset_json.append(datum_json)
    line_cnt+=1

with open(args.save_path, 'w') as f:
  for datum in dataset_json:
    json.dump(datum, f)
    f.write('\n')
