from datasets import concatenate_datasets, load_dataset
from .text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)
import json

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



### --- sampling functions ---
def sample_portion_of_data(ds):
  sample_size = 200_000
  pn_ds = ds.filter(lambda example: example["ds"] == 'pn_summary').shuffle(seed=30).select(range(0, sample_size))
  ds = ds.filter(lambda example: example["ds"] != 'PNLPhub/DigiMag' and 
                                 example["ds"] != 'PNLPhub/Persian-News' and
                                 example["ds"] != 'pn_summary')

  new_ds = concatenate_datasets([ds, pn_ds])
  return new_ds

def select_zs_ds(ds):
   return ds.filter(lambda x: x['type'] == 'zs' and x['ds'] == 'persiannlp/parsinlu_sentiment')

def select_ds(ds):
   return ds.filter(lambda x: x['ds'].startswith('pn_summary'))

def sample_ds_with_acc(raw_data, ds_name):
  def map_fn(ex):
    ds_meta_data = load_meta_data()[ds_name]
    for task in ds_meta_data: 
        if ex['template'] == task['template'] and \
           ex['ds'] == ds_name and \
           'Accuracy' in task['metrics']:

           return ex
            
  


  return raw_data.filter(map_fn)

