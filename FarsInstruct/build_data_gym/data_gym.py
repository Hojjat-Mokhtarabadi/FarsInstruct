from promptsource.templates import DatasetTemplates
from datasets import load_dataset
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import os
import json

class DataGym:
    """
    Apply template on datasets according to they specified type (zero-shot or few-shot)
    """
    def __init__(self, dataset_name:str, subset_name: str, template_name: str, split: str, min_samples: int = 0):        
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        self.template_name = template_name

        try:
            self.data = load_dataset(self.dataset_name, split=split)
        except:
            self.data = load_dataset("../wiki_summary", split=split)
            
        if min_samples != 0:
            self.data = self.data.shuffle(seed=1531).select(range(0, min(min_samples, len(self.data))))

        self.template = DatasetTemplates(self.dataset_name, self.subset_name)[template_name]

    def build_zs_gym(self):
        inputs = []; outputs = [] 
        for example in tqdm(self.data, total=len(self.data)):
            result = self.template.apply(example)
            txt = '\n'.join(result[0].split('\n\n'))          
            inputs.append(txt)
            outputs.append(result[1])
            
        result_dict = {'inputs': inputs, 'outputs': outputs, 
                       'ds': self.dataset_name, 'template': self.template_name}
        
        save_data(result_dict, self.dataset_name, self.template_name, self.split, self.subset_name)

        return
    
    def build_fs_gym(self, shots):
        inputs = []; outputs = []

        def remove_instruction(x):
            splt_text = x.split('\n\n')
            return '\n'.join(splt_text[1:])

        for i in tqdm(range(0, (len(self.data) - shots - 1), shots)):
            result_fs = ""
            output = ""
            for idx in range(i, i + shots):
                result = self.template.apply(self.data[idx])  
                output = result[1]
                if idx == i: # instruct line
                    input_ = result[0]
                    result_fs += (input_ + output + '\n\n')

                elif idx == (i + shots - 1): # last line without instruction
                    input_wo_instruct = remove_instruction(result[0])
                    result_fs += (input_wo_instruct + '\n')

                else: # body line without instruction
                    input_wo_instruct = remove_instruction(result[0])
                    result_fs += (input_wo_instruct + output + '\n\n')


            inputs.append(result_fs)
            outputs.append(output)

        result_dict = {'inputs': inputs, 'outputs': outputs, 
                       'ds': self.dataset_name, 'template': self.template_name}
        
        save_data(result_dict, self.dataset_name, self.template_name, self.split, self.subset_name)

        return


def save_data(result, dataset_name, template_name, split, subset_name = None):
    df = pd.DataFrame.from_dict(result)
    if subset_name == None:
        dir = f"data/{dataset_name}/{split}"
    else: 
        dir = f"data/{dataset_name}_{subset_name}/{split}"
    if os.path.exists(dir):
        df.to_csv(f"{dir}/{template_name}_{split}.csv", mode='w+')
    else:
        os.makedirs(dir)
        df.to_csv(f"{dir}/{template_name}_{split}.csv", mode='w+')

   
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template_name', type=str, required=True)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()



