from datasets import load_dataset
from data.hf_dataset import load_hf_ds_from_csv
import pandas as pd
from .utils import sample_portion_of_data, normalization


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
            self.raw_dataset = load_hf_ds_from_csv(self.split, self.stream)
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)

        # rather than the whole dataset select a portion of it
        self.raw_dataset = sample_portion_of_data(self.raw_dataset)

    def preprocess(self, example) -> str: 
        prompt = normalization(example['inputs']) + '<|startoftext|>' + normalization(example['outputs'])
        return prompt


    def encode(self, example):
        prompt = self.preprocess(example)
        new_prompt = '<s>' + prompt + '</s>'
        return self.tokenizer(new_prompt, truncation=True, max_length=self.max_len, padding=True, return_tensors='pt')
    

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


