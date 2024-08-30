from datasets import concatenate_datasets, Dataset
from .text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)
import json
from tqdm import tqdm
import random

def normalization(text):   
    text = text.replace("[n]", " ")
    text = "".join(map_to_persian(char) for char in text)
    # Split the text into sentences
    # snt = ''
    # for sentence in split_into_sentences(text):
    #     # Remove the remaining punctuation
    #     # sentence = patterns["ELIMINATE"].sub(" ", sentence)
    #     # Making sure there's a space after each comma
    #     sentence = patterns["COMMA"].sub("، ", sentence)
    #     # Multiple spaces into one
    #     # sentence = patterns["TOO_MANY_SPACES"].sub(" ", sentence)
    #     # Strip the leading and the trailing white spaces
    #     # sentence = sentence.strip()
    #     # Remove the spaces before punctuations
    #     # sentence = patterns["NO_SPACE_BEFORE"].sub("", sentence)

    #     snt += sentence
    return text

def load_meta_data():
  with open('data/metadata.json', 'r', encoding='utf-8') as f:
      meta_data = json.load(f)

  return meta_data 


### --- sampling functions ---
def sample_dataset(raw_data, ds_name):  
   #min_chunk = 2000
   ds_list = []
   temp_list = {
           "title_generate": 120, 
           "question_or_answer_catg": 120,
           "generate_question_wrt_answer": 120,
           "summarize_the_article": 120,
           "what_category_it_belongs_to": 120,
           "gen_q_with_long_short_ans": 120,
           "write_article_summary": 120,
           "in_which_categ_would_it_go": 120,
           "choose_category": 120,
           "general_answer": 120,
           "review_aspect": 120,
           "relation": 120,
           "which_polarity": 120,
           "rewrite_order": 120,
           "question_or_answer": 120,
           "review_category": 120,
           "find_person": 120,
           "find_answer": 120,
           "question_context": 120,
           "return_possibility": 120,
           "summarize_article": 120,
           "provide_english": 120,
           "rate": 120,
           "title_to_text": 120,
           "answer_question": 120,
           "translate_english": 120,
           "similar": 15_000
           }
   for ds in ds_name:
    #if ds == "SajjadAyoubi/persian_qa":
    #    min_chunk = 10_000
    #else:
    #    min_chunk = 100
   
    for k, v in temp_list.items():
        raw_data_filterd = raw_data.filter(lambda ex: ex["ds"] == ds and ex["template"] == k)
        raw_data_filterd = raw_data_filterd.shuffle(seed=random.randint(1, 2000)).select(range(0, min(10_000, len(raw_data_filterd))))
        ds_list.append(raw_data_filterd)
     
    '''
    min_chunk = 11_000
    raw_data_filterd = raw_data.filter(lambda ex: ex["ds"] == ds)
    raw_data_filterd = raw_data_filterd.shuffle(seed=random.randint(1, 2000)).select(range(0, min(min_chunk, len(raw_data_filterd))))
    
    ds_list.append(raw_data_filterd)
    '''
    '''
    raw_data_filterd = raw_data.filter(lambda ex: ex["ds"] == ds)
    temps = raw_data_filterd.unique("template")
    for temp in temps:
        if temp == "title_generate":
            min_chunk = 100
        elif temp == "question_or_answer_catg":
            min_chunk = 100
        elif temp == "generate_question_wrt_answer":
            min_chunk = 100
        elif temp == "summarize_the_article":
            min_chunk = 100
        elif temp == "what_category_it_belongs_to":
            min_chunk = 10_000
        else:
            min_chunk = 0

        raw_data_filterd_temp = raw_data_filterd.filter(lambda ex: ex["template"] == temp)
        #samp_size = (min_chunk // len(temps))
        raw_data_filterd_temp = raw_data_filterd_temp.shuffle(seed=random.randint(1, 2000)).select(range(0, min(128, len(raw_data_filterd_temp))))
        ds_list.append(raw_data_filterd_temp)
    '''

   return concatenate_datasets(ds_list)
