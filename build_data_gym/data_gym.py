from promptsource.templates import DatasetTemplates
from datasets import load_dataset
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import os
import re


def save_data(result, dataset_name, template_name, split):
    df = pd.DataFrame.from_dict(result)
    dir = f"data/{dataset_name}"
    if os.path.exists(dir):
        df.to_csv(f"{dir}/{template_name}_{split}.csv")
    else:
        os.makedirs(dir)
        df.to_csv(f"{dir}/{template_name}_{split}.csv")


class DataGym:
    """
    Apply template on datasets according to they specified type (zero-shot or few-shot)
    """
    def __init__(self, dataset_name:str, template_name: str, split: str, 
                       type: str, shots: int = 1):
        
        self.dataset_name = dataset_name
        self.data = load_dataset(self.dataset_name, split=split)
        self.shots = shots
        self.sample_range = len(self.data) // self.shots
        self.type = type
        self.split = split
        self.template_name = template_name

        self.template = DatasetTemplates(self.dataset_name)[template_name]

    def __call__(self):
        if self.type == 'zs':
            return self._build_zs_gym()
        elif self.type == 'fs':
            return self._build_fs_gym() 

    def _build_zs_gym(self):
        inputs = []; outputs = []    
        for example in tqdm(self.data, total=len(self.data)):
            result = self.template.apply(example)
            inputs.append(result[0])
            outputs.append(result[1])

        result_dict = {'inputs': inputs, 'outputs': outputs, 'type': self.type, 'ds': self.dataset_name}
        save_data(result_dict, self.dataset_name, self.template_name, self.split)

        return

    
    def _build_fs_gym(self):
        inputs = []; outputs = []

        def remove_instruction(x):
            space = re.compile("\\s+")
            splt_text = x.split('\n')
            snt_list = []
            for snt in splt_text:
                snt_list.append(space.sub(" ", snt))

            return '\n'.join(snt_list[1:])

        for i in tqdm(range(0, (len(self.data) - self.shots - 1), self.shots)):
            result_fs = ""
            output = ""
            for idx in range(i, i + self.shots):
                result = self.template.apply(self.data[idx])  
                output = result[1]

                if idx == i:
                    input_ = result[0]
                    result_fs += (input_ + output + '\n')

                elif idx == (i + self.shots - 1):
                    input_wo_instruct = remove_instruction(result[0])
                    result_fs += (input_wo_instruct + '\n')

                else:
                    input_wo_instruct = remove_instruction(result[0])
                    result_fs += (input_wo_instruct + output + '\n')

                    
            inputs.append(result_fs)
            outputs.append(output)

        result_dict = {'inputs': inputs, 'outputs': outputs, 'type': self.type, 'ds': self.dataset_name}
        save_data(result_dict, self.dataset_name, self.template_name, self.split)

        return

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template_name', type=str, required=True)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--type', type=str, default='zs', required=True)
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()



