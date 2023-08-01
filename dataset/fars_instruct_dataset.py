from datasets import load_dataset
from data.hf_dataset import load_hf_ds_from_csv
from tqdm import tqdm
import pandas as pd
from .text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)


class FarsInstructDataset:
    def __init__(self, tokenizer, max_len: int, split: str, stream: bool, dataload_mode: str, dataset_path: str):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.stream = stream

        if dataload_mode == 'local':
            self.raw_dataset = load_hf_ds_from_csv(self.split, self.stream)
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)


    def preprocess(self, example) -> str: 
        def normalization(text):   
            # normalizer = Normalizer(persian_numbers=False)
            # return normalizer.normalize(text)
            text = text.replace("[n]", "\n")
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
        
        prompt = normalization(example['inputs']) + '<|startoftext|>' + normalization(example['outputs'])
        return prompt


    def encode(self, example):
        prompt = self.preprocess(example)
        new_prompt = '<s>' + prompt + '</s>'
        return self.tokenizer(new_prompt, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
    

    def get_tokenized_data(self, in_torch_format: bool = True):
        tokenized_data = self.raw_dataset.map(self.encode, batched=False)
        tokenized_data = tokenized_data.remove_columns(['inputs', 'outputs', 'type', 'ds'])
        if in_torch_format:
            return tokenized_data.with_format('torch')
        else:
            return tokenized_data

        
        

if __name__ == "__main__":
    df = pd.DataFrame({
        'txt': ["""با توجه به سوال یک جواب کوتاه و یک جواب بلند بنویس

سوال: باشگاه هاکی ساوتهمپتون چه نام دارد؟
جواب کوتاه:"""]
    })


