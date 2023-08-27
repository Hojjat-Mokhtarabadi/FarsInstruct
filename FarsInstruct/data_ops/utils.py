from datasets import concatenate_datasets, load_dataset
from .text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)

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

