from datasets import load_dataset
import pandas as pd
from .utils import *
from .paths import DATA_FILES


class FarsInstructDataset:
    def __init__(self, tokenizer, max_len: int, split: str, stream: bool, datasets: str,
                 dataload_mode: str, dataset_path: str, instruction_template: str, **kwargs):
        """
        FarsInstruct Dataset
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.stream = stream
        self.meta_data = load_meta_data()
        self.instruction_template = instruction_template
        self.datasets = datasets

        # 'local' model loads data from the local csv file, 'hub' downloads it.
        if dataload_mode == 'local':
            # each model accepts different insturction template. select each based on config file.
            self.raw_dataset = load_dataset('csv', data_files=DATA_FILES[instruction_template], split=split, streaming=self.stream)
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)

        # rather than the whole dataset select a portion of it
        ds_list = self.datasets.split(',')
        print(f"Training datasets: {ds_list}")

        self.raw_dataset = sample_dataset(self.raw_dataset, ds_list)
       
    def preprocess(self, example, idx) -> str:
        prompt = normalization(example['inputs'][idx]) +  normalization(example['outputs'][idx])        
        return prompt
    
    
    def pretraining_encode_fn(self, example): 
        """
        preprocess the inputs example and tokenize it. 
         - features.keys() --> (input_ids, attention_mask)
        """
        bs = len(example['inputs'])
        batch_features = []
        for i in range(bs):
            prompt = self.preprocess(example, i)
            new_prompt = '<s>' + prompt + '</s>'
            batch_features.append(self.tokenizer(new_prompt, truncation=True, max_length=self.max_len, 
                                    padding='max_length', return_tensors='pt'))

        features = {'input_ids': [], 'attention_mask': []}
        for feature in batch_features:
            for k, v in feature.items():
                features[k].append(v)
        
        return features


    def get_tokenized_data(self, in_torch_format: bool = True):
        tokenized_data = self.raw_dataset.map(self.pretraining_encode_fn, batched=True, 
                                                remove_columns=['inputs', 'outputs', 
                                                                'type', 'ds', 'template'])
        
        if in_torch_format:
            return tokenized_data.with_format('torch')
        else:
            return tokenized_data
        

    def remove_mid_dim(tokenized_data):
        new_tokenized_data = {}
        for c in tokenized_data:
            new_tokenized_data[c] = [x[0] for x in tokenized_data[c]]
        return new_tokenized_data


if __name__ == "__main__":
    df = pd.DataFrame({
        'txt': ["""با توجه به سوال یک جواب کوتاه و یک جواب بلند بنویس

سوال: باشگاه هاکی ساوتهمپتون چه نام دارد؟
جواب کوتاه:"""]
    })


