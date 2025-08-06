from tokenizers.processors import TemplateProcessing
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
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream, cache_dir="/mnt/beegfs/wrkdir/u111187/Hojjat_Workstation/farsinstruct_data")

        if self.datasets != 'all':
            # rather than the whole dataset select a portion of it
            ds_list = self.datasets.split(',')
            print(f"Training datasets: {ds_list}")
            self.raw_dataset = sample_dataset(self.raw_dataset, ds_list)

        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=self.tokenizer.bos_token + " $A " + self.tokenizer.eos_token,
        special_tokens=[(self.tokenizer.eos_token, self.tokenizer.eos_token_id), 
                        (self.tokenizer.bos_token, self.tokenizer.bos_token_id)],)


       
    def preprocess(self, example) -> str:
        prompt = normalization(example)

        if self.instruction_template == 'llama':
            prompt = f"[INST]{prompt}[/INST]"

        elif self.instruction_template == 'hooshvare':
            prompt = f"{prompt} <startoftext>"   

        elif self.instruction_template == 'mgpt':
            prompt = f"{prompt} [INST]"     

        elif self.instruction_template == "ava":
            prompt = f"{prompt}"
            # prompt = "### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"

        return prompt
    
    
    def pretraining_encode_fn(self, example): 
        """
        preprocess the inputs example and tokenize it. 
         - features.keys() --> (input_ids, attention_mask)
        """
        input_ = self.preprocess(example['inputs'])
        target = example['outputs']
        prompt = f"### Instruction: {input_} \n ### Response: {normalization(target)}"        
        features = self.tokenizer(prompt, truncation=False, max_length=self.max_len, padding='max_length')
        tokenized_input = self.tokenizer(target)
        
        return features


    def get_tokenized_data(self, in_torch_format: bool = True):
        special_token_dict = self.tokenizer.special_tokens_map
        self.tokenizer.add_special_tokens(special_token_dict)
        # self.tokenizer.add_special_tokens({"pad_token" : "<pad>"})
        
        tokenized_data = self.raw_dataset.map(self.pretraining_encode_fn, batched=False, remove_columns=self.raw_dataset.column_names)
        tokenized_data = tokenized_data.filter(lambda example: len(example["input_ids"]) <= self.max_len)
        
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


