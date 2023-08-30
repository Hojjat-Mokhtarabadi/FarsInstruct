from datasets import load_dataset
import pandas as pd
import json
from .utils import *
from .paths import DATA_FILES


class FarsInstructDataset:
    def __init__(self, tokenizer, max_len: int, split: str, stream: bool, dataload_mode: str, dataset_path: str):
        """
        FarsInstruct Dataset
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.stream = stream
        self.meta_data = load_meta_data()

        if dataload_mode == 'local':
            self.raw_dataset = load_dataset('csv', data_files=DATA_FILES, split=split, streaming=self.stream)
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)

        # rather than the whole dataset select a portion of it
        self.raw_dataset = sample_ds_with_acc(self.raw_dataset, 'pn_summary')

       
    def preprocess(self, example) -> str:
        prompt = normalization(example['inputs']) + '<|startoftext|>' + normalization(example['outputs'])
        
        return prompt


    def encode_fn(self, example): 
        """
        preprocess the inputs example and tokenize it. 
         - features.keys() --> (input_ids, attention_mask, targets)
         - if answer_choice is null, replace it with <emp> token.
        """
        prompt = self.preprocess(example)
        new_prompt = '<s>' + prompt + '</s>'
        features = self.tokenizer(new_prompt, truncation=True, max_length=self.max_len, 
                                  padding='max_length', return_tensors='pt')

        ds_meta_data = self.meta_data[example['ds']]
        answer_choices = None
        for task in ds_meta_data: 
            if example['template'] == task['template']:
                if task['choice_in_temp']:
                    answer_choices = task['ans_choice']
                else:
                    answer_choices = ['<emp>']
                break
        
        target_texts = []
        answer_choices_texts = []
        # add 'targets' key into the 'features' dictionary
        tokenized_targets = [
                self.tokenizer(
                    ans_choi,
                    # padding is on the right here.
                    padding=False,
                    max_length=self.max_len,
                    truncation=True,
                )
                for ans_choi in answer_choices
            ]
        
        features['targets'] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]
        
        return features

    def get_tokenized_data(self, in_torch_format: bool = True):
        tokenized_data = self.raw_dataset.map(self.encode_fn, batched=False, 
                                                remove_columns=['inputs', 'outputs', 
                                                                'type', 'ds', 'template'])
        
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


