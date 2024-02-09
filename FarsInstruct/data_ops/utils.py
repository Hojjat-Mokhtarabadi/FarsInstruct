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
def sample_portion_of_data(raw_data):
  zs_sample_size = 100_000
  fs_sample_size = 100_000
  pn_ds_zs = raw_data.filter(lambda example: example["ds"] == 'pn_summary' and example['type'] == 'zs').shuffle(seed=30).select(range(0, zs_sample_size))
  pn_ds_fs = raw_data.filter(lambda example: example["ds"] == 'pn_summary' and example['type'] == 'fs').shuffle(seed=30).select(range(0, fs_sample_size))
  raw_data = raw_data.filter(lambda example: example["ds"] != 'PNLPhub/DigiMag' and 
                                 example["ds"] != 'PNLPhub/Persian-News' and
                                 example["ds"] != 'pn_summary' and
                                 example['ds'] != 'PNLPhub/digikala-sentiment-analysis')

  new_ds = concatenate_datasets([raw_data, pn_ds_zs, pn_ds_fs])
  return new_ds


def sample_data_for_eval(raw_data, ds_name, metric):
   def map_fn(ex):
      for ds in ds_name:
         ds_meta_data = load_meta_data()[ds]
         for task in ds_meta_data: 
            if ex['template'] == task['template'] and \
               ex['ds'] == ds and \
               metric in task['metrics']:
               
               return ex

   return raw_data.filter(map_fn)

def exclude_datasets(raw_data, ds_name):
   ds = raw_data.filter(lambda ex: ex["ds"] not in ds_name)

   return ds


def sample_dataset(raw_data, ds_name): 
   if "pn_summary" in ds_name: 
      zs_sample_size = 50_000
      fs_sample_size = 50_000
      pn_ds_zs = raw_data.filter(lambda example: example["ds"] == 'pn_summary' and example['type'] == 'zs').shuffle(seed=30).select(range(0, zs_sample_size))
      pn_ds_fs = raw_data.filter(lambda example: example["ds"] == 'pn_summary' and example['type'] == 'fs').shuffle(seed=30).select(range(0, fs_sample_size))

      ds_name.remove("pn_summary")
      # ds = raw_data.filter(lambda ex: ex["ds"] in ds_name)
      #ds = raw_data.filter(lambda ex: ex["ds"] in ds_name and ex['type'] == 'zs')

      # return concatenate_datasets([ds, pn_ds_zs, pn_ds_fs])
   
   if "PNLPhub/snappfood-sentiment-analysis" in ds_name: 
      zs_sample_size = 45_000
      fs_sample_size = 45_000
      pn_ds_zs = raw_data.filter(lambda example: example["ds"] == 'PNLPhub/snappfood-sentiment-analysis' and example['type'] == 'zs').shuffle(seed=30).select(range(0, zs_sample_size))
      pn_ds_fs = raw_data.filter(lambda example: example["ds"] == 'PNLPhub/snappfood-sentiment-analysis' and example['type'] == 'fs').shuffle(seed=30).select(range(0, fs_sample_size))

      ds_name.remove("PNLPhub/snappfood-sentiment-analysis")
      
      ds = raw_data.filter(lambda ex: ex["ds"] in ds_name)
      #ds = raw_data.filter(lambda ex: ex["ds"] in ds_name and ex['type'] == 'zs')

      return concatenate_datasets([ds, pn_ds_zs, pn_ds_fs])
   
   else: 
      return raw_data.filter(lambda ex: ex["ds"] in ds_name) 

