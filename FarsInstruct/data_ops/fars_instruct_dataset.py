from datasets import load_dataset
import pandas as pd
from .utils import *


class FarsInstructDataset:
    def __init__(self, tokenizer, max_len: int, split: str, stream: bool, datasets: str,
                 dataload_mode: str, dataset_path: str, instruction_template: str, shots: int, **kwargs):
        """
        FarsInstruct Dataset
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.stream = stream
        self.instruction_template = instruction_template
        self.datasets = datasets
        self.shots = shots

        # 'local' model loads data from the local csv file, 'hub' downloads it.
        if dataload_mode == 'local':
            # each model accepts different insturction template. select each based on config file.
            DATA_FILES = {'train': dataset_path}
            self.raw_dataset = load_dataset('csv', data_files=DATA_FILES, split=split, streaming=self.stream)
            
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)

        if self.datasets != 'all':
            # rather than the whole dataset select a portion of it
            ds_list = self.datasets.split(',')
            print(f"Training datasets: {ds_list}")

            self.raw_dataset = sample_dataset(self.raw_dataset, ds_list)

       
    def preprocess(self, example) -> str:
        prompt = normalization(example)

        if self.instruction_template == 'llama':
            prompt = f"[INST]{prompt}[/INST]"

        elif self.instruction_template == 'hooshvare':
            prompt = f"{prompt} <startoftext>"   

        elif self.instruction_template == 'mgpt':
            prompt = f"{prompt} [INST]"     

        elif self.instruction_template == "alpaca":
            prompt = f"<|im_start|>{prompt}<|im_end|>\n<|im_start|>"

        return prompt
    
    
    def pretraining_encode_fn(self, example): 
        """
        preprocess the inputs example and tokenize it. 
         - features.keys() --> (input_ids, attention_mask)
        """
        input_ = self.preprocess(example['inputs'])
        target = example['outputs']
        prompt = input_ + normalization(target)+"<|im_end|><|end_of_text|>"
        features = self.tokenizer(prompt, truncation=True, max_length=self.max_len, padding='max_length')
        
        return features


    def get_tokenized_data(self, in_torch_format: bool = True):
        special_token_dict = self.tokenizer.special_tokens_map
        self.tokenizer.add_special_tokens(special_token_dict)
        
        tokenized_data = self.raw_dataset.map(self.pretraining_encode_fn, batched=False, remove_columns=self.raw_dataset.column_names)
        
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


