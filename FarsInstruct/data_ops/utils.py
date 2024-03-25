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
        # sentence = patterns["NO_SPACE_BEFORE"].sub("", sentence)

        snt += sentence
    return snt

def load_meta_data():
  with open('data/metadata.json', 'r', encoding='utf-8') as f:
      meta_data = json.load(f)

  return meta_data 


### --- sampling functions ---
def sample_dataset(raw_data, ds_name):  
   min_chunk = 12_000
   ds_list = []
   for ds in ds_name:
    raw_data_filterd = raw_data.filter(lambda ex: ex["ds"] == ds) 
    # raw_data_filterd = raw_data_filterd.shuffle(seed=30).select(range(0, min(min_chunk, len(raw_data_filterd))))
    ds_list.append(raw_data_filterd)
    

   return concatenate_datasets(ds_list)
