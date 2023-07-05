from promptsource.templates import DatasetTemplates
from datasets import load_dataset
from argparse import ArgumentParser
import pandas as pd


class FarsTail:
    def __init__(self, template_name: str, split: str, type: str, shots: int = 1):
        self.dataset_name = "PNLPhub/FarsTail"
        self.data = load_dataset(self.dataset_name, split=split)
        self.shots = shots
        self.sample_range = len(self.data) // self.shots
        self.type = type

        self.template = DatasetTemplates(self.dataset_name)[template_name]

    def __call__(self):
        if self.type == 'zs':
            return self._build_zs_gym()
        elif self.type == 'fs':
            return self._build_fs_gym() 

    def _build_zs_gym(self):
        inputs = []; outputs = []    
        for idx, example in enumerate(self.data):
            result = self.template.apply(example)
            inputs.append(result[0])
            outputs.append(result[1])

        return {'inputs': inputs, 
                'outputs': outputs, 
                'type': self.type}
    
    def _build_fs_gym(self):
        inputs = []; outputs = []
        remove_instruction = lambda x: '\n'.join(x.split('\n')[3:])

        for i in range(0, (len(self.data) - self.shots - 1), self.shots):
            result_fs = ""
            output = ""
            for idx in range(i, i + self.shots):
                result = self.template.apply(self.data[idx])  
                output = result[1]

                if idx == 0:
                    input = result[0]
                    result_fs += (input + output + '\n')

                elif idx == (i + self.shots - 1):
                    input_wo_instruct = remove_instruction(result[0])
                    result_fs += (input_wo_instruct + '\n')

                else:
                    input_wo_instruct = remove_instruction(result[0])
                    result_fs += (input_wo_instruct + output + '\n')

                    
            inputs.append(result_fs)
            outputs.append(output)

        return {'inputs': inputs, 
                'outputs': outputs, 
                'type': self.type}
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template_name', type=str, required=True)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--type', type=str, default='zs', required=True)
    args = parser.parse_args()

    ftail = FarsTail(args.template_name, args.split, args.type, args.shots)
    df = pd.DataFrame.from_dict(ftail())
    df.to_csv(f"../data/farstail/{args.template_name}_{args.split}.csv")



