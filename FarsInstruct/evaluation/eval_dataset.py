from datasets import load_dataset
import json
from FarsInstruct.data_ops.paths import DATA_FILES

class FarsInstructEvalDataset:
    def __init__(self, tokenizer, max_len: int, instruction_template: str, **kwargs):
        self.ans_choices = []
        self.tokenizer = tokenizer       
        # each model accepts different instruction template, select each based on config file.
        self.ds = load_dataset('csv', data_files=DATA_FILES[instruction_template], split='validation')    
        self.max_len = max_len
        self.extra_cols = ['inputs', 'outputs', 'ds', 'type', 'template']
        
    def load_meta_data(self):
        with open('data/metadata.json', 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        return meta_data

    def _get_ans_choices(self, ds_name, temp_name):
        meta_data = self.load_meta_data()
        for temp in meta_data[ds_name]:
            if temp['template'] == temp_name and temp['choice_in_temp'] is True:
                self.ans_choices = temp['ans_choice']
                return

    def _likelihood_preprocess_fn(self, ex):
        bs = len(ex["inputs"])
        num_choices = len(self.ans_choices)
        input_txt = [context for context in ex["inputs"]]
        target_txt = [target for target in ex["outputs"]]
        answer_choices = [self.ans_choices for _ in ex["inputs"]]

        tokenized_inputs = self.tokenizer(input_txt, truncation=True, max_length=self.max_len)
        tokenized_targets = [
            self.tokenizer(
                ans_choi,
                truncation=True,
            )
            for ans_choi in answer_choices
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
            answer_choices[idx].index(t)
            for idx, t in enumerate(target_txt)
        ]

        return features
    
    def _generation_preprocess_fn(self, ex):
        inputs = ex['inputs']
        targets = ex['outputs']
        features = self.tokenizer(inputs, truncation=True, max_length=self.max_len, padding='max_length')
        labels = self.tokenizer(text_target=targets, truncation=True, max_length=self.max_len, padding='max_length')

        features['labels'] = labels['input_ids']
        return features
    
    
    def get_tokenized_data(self, ds_name: str, temp_name: str, multiple_choice: bool):
        if multiple_choice:
            self._get_ans_choices(ds_name, temp_name)
            preprocess_fn = self._likelihood_preprocess_fn
        else:
            preprocess_fn = self._generation_preprocess_fn

        self.ds = self.ds.filter(lambda x: x['ds'] == ds_name and x['template'] == temp_name)
        return self.ds.map(preprocess_fn, batched=True, remove_columns=self.extra_cols)



if __name__ == "__main__":
    from transformers import AutoTokenizer

    checkpoint = 'HooshvareLab/gpt2-fa'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="right", pad_token='<pad>')

    ds = FarsInstructEvalDataset(tokenizer, 20, 'hooshvare')
    encoded_ds = ds.get_tokenized_data('PNLPhub/FarsTail', 'label_to_hypothesis_zs', multiple_choice=False)
    encoded_ds2 = ds.get_tokenized_data('PNLPhub/FarsTail', 'can_you_infer_zs', multiple_choice=True)

    # print(encoded_ds)
    # print(encoded_ds['labels'][:10])
    print(encoded_ds2)