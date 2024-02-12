from datasets import concatenate_datasets, Dataset
from .text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)
import json
from tqdm import tqdm

def normalization(text):   
    text = text.replace("[n]", " ")
    text = "".join(map_to_persian(char) for char in text)
    # Split the text into sentences
    snt = ''
    for sentence in split_into_sentences(text):
        # Remove the remaining punctuation
        # sentence = patterns["ELIMINATE"].sub(" ", sentence)
        # Making sure there's a space after each comma
        sentence = patterns["COMMA"].sub("، ", sentence)
        # Multiple spaces into one
        # sentence = patterns["TOO_MANY_SPACES"].sub(" ", sentence)
        # Strip the leading and the trailing white spaces
        sentence = sentence.strip()
        # Remove the spaces before punctuations
        sentence = patterns["NO_SPACE_BEFORE"].sub("", sentence)

        snt += sentence
    return snt

def load_meta_data():
  with open('data/metadata.json', 'r', encoding='utf-8') as f:
      meta_data = json.load(f)

  return meta_data 


# def build_fewshot_gym(raw_dataset, shots):
#    inputs = []; outputs = []

#    def remove_instruction(x):
#       splt_text = x.split('\n\n')
#       return '\n'.join(splt_text[1:])

#    for i in range(0, (len(raw_dataset) - shots - 1), shots):
#       result_fs = ""
#       output = ""
#       for idx in range(i, i + shots):
#             output = raw_dataset['outputs'][idx]
#             input_ = raw_dataset['inputs'][idx]
#             if idx == i:
#                result_fs += (input_ + output + '\n')

#             elif idx == (i + shots - 1):
#                input_wo_instruct = remove_instruction(input_)
#                result_fs += (input_wo_instruct + '\n')

#             else:
#                input_wo_instruct = remove_instruction(input_)
#                result_fs += (input_wo_instruct + output + '\n')


#       inputs.append(result_fs)
#       outputs.append(output)
            
#    return Dataset.from_dict({'inputs': inputs, 'outputs': outputs})



### --- sampling functions ---
def sample_dataset(raw_data, ds_name):  
   min_chunk = 50_000
   ds_list = []
   for ds in ds_name:
    raw_data_filterd = raw_data.filter(lambda ex: ex["ds"] == ds) 
    raw_data_filterd = raw_data_filterd.shuffle(seed=30).select(range(0, min(min_chunk, len(raw_data_filterd))))
    ds_list.append(raw_data_filterd)
    

   return concatenate_datasets(ds_list)

   # if shots > 1:
   #    print('Building fewshot gym...')
   #    dataset_lst = []
   #    for ds in tqdm(ds_name):
   #       raw_dataset = raw_data.filter(lambda ex: ex["ds"] == ds)
   #       dataset_lst.append(build_fewshot_gym(raw_dataset, shots))

   #    return concatenate_datasets(dataset_lst)

