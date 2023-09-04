from datasets import load_dataset
import pandas as pd
from .utils import *
from .paths import DATA_FILES


class FarsInstructDataset:
    def __init__(self, tokenizer, max_len: int, split: str, stream: bool, 
                 dataload_mode: str, dataset_path: str, **kwargs):
        """
        FarsInstruct Dataset
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.stream = stream
        self.meta_data = load_meta_data()

        # 'local' model loads data from the local csv file, 'hub' downloads it.
        if dataload_mode == 'local':
            self.raw_dataset = load_dataset('csv', data_files=DATA_FILES, split=split, streaming=self.stream)
        elif dataload_mode == 'hub':
            self.raw_dataset = load_dataset(dataset_path, split=self.split, streaming=self.stream)

        # rather than the whole dataset select a portion of it
        # self.raw_dataset = sample_data_for_eval(self.raw_dataset, metric='Accuracy', 
        #                                ds_name=['PNLPhub/digikala-sentiment-analysis', 
        #                                         'PNLPhub/snappfood-sentiment-analysis'])
        self.raw_dataset = sample_portion_of_data(self.raw_dataset)

       
    def preprocess(self, example, idx) -> str:
        prompt = normalization(example['inputs'][idx]) + '<|startoftext|>' + normalization(example['outputs'][idx])
        
        return prompt
    
    def encode_fn_based_on_t0(self, examples):
        bs = len(examples['inputs'])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            input, target = normalization(examples['inputs'][i]) + '<|startoftext|>', examples['outputs'][i]
            
            ds_meta_data = self.meta_data[examples['ds'][i]]
            ex_answer_choices = None
            for task in ds_meta_data: 
                if examples['template'][i] == task['template']:
                    if task['choice_in_temp']:
                        ex_answer_choices = task['ans_choice']
                    else:
                        ex_answer_choices = ['<emp>']
                    break
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        tokenized_inputs = self.tokenizer(
            input_texts,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        )

        tokenized_targets = [
            self.tokenizer(
                ans_choi,
                # padding is on the right here.
                padding=False,
                max_length=self.max_len,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }

        features["labels"] = [
            tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]

        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]

        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
            ]
        
        return features


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


if __name__ == "__main__":
    df = pd.DataFrame({
        'txt': ["""با توجه به سوال یک جواب کوتاه و یک جواب بلند بنویس

سوال: باشگاه هاکی ساوتهمپتون چه نام دارد؟
جواب کوتاه:"""]
    })


