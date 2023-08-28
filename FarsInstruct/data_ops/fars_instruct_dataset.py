from datasets import load_dataset
import pandas as pd
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

        if dataload_mode == 'local':
            self.raw_dataset = load_dataset('csv', data_files=DATA_FILES, split=split, streaming=self.stream)
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)

        # rather than the whole dataset select a portion of it
        #self.raw_dataset = sample_portion_of_data(self.raw_dataset)

    def preprocess(self, example) -> str:
        prompt = normalization(example['inputs']) + '<|startoftext|>' + normalization(example['outputs'])
        
        return prompt


    def encode_fn(self, example): 
        # preprocess the inputs example and tokenize it
        # 'self.tokenizer' return a dictionary with 'input_ids' and 'attention_masks' as keys,
        # create a features dict with the same keys
        prompt = self.preprocess(example)
        new_prompt = '<s>' + prompt + '</s>'
        features = self.tokenizer(new_prompt, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')

        # add 'targets' key into the 'features' dictionary
        answer_choices = eval(example['ans_choices']) if example['ans_choices'] != 'emp' else ['<emp>']
        features['targets'] = [
                self.tokenizer(
                    ans_choi,
                    # padding is on the right here.
                    padding=False,
                    max_length=self.max_len,
                    truncation=True,
                    return_tensor='pt'
                )
                for ans_choi in answer_choices
            ]
        
        return features

    def get_tokenized_data(self, in_torch_format: bool = True):
        tokenized_data = self.raw_dataset.map(self.encode_fn, batched=False, 
                                                remove_columns=['inputs', 'outputs', 'type', 
                                                                'ds', 'ans_choices'])
        
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


