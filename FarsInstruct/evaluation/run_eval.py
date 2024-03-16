import json
import torch
import numpy as np
from tqdm import tqdm
import warnings
import evaluate
from transformers import AutoTokenizer
from argparse import ArgumentParser
from accelerate import Accelerator
from torch.utils.data import DataLoader 

from transformers import DataCollatorWithPadding
from hazm import sent_tokenize

from prettytable import PrettyTable

from FarsInstruct.evaluation.data_collator import DataCollatorForMultipleChoice
from FarsInstruct.evaluation.eval_dataset import FarsInstructEvalDataset
from FarsInstruct.evaluation.model import DecoderModel, load_causal_model
from FarsInstruct.evaluation.temp_list import TEMP_LIST

from FarsInstruct.utils import EvaluationArgs, DatasetArgs, load_yml_file

#! ignore sourceTensor.clone().detach() warning
warnings.filterwarnings("ignore", category=UserWarning)

class LMEvaluation:
    """
    Evaluation
    """
    def __init__(self, configs, tokenizer: AutoTokenizer, split: str):
        self.accelerator = Accelerator(cpu=False)
        self.eval_args = EvaluationArgs(**configs['evaluation_args'])
        self.data_args = DatasetArgs(**configs['dataset_args'])
        self.configs = configs
        self.split = split
        self.shots = self.eval_args.shots
        self.run_name = configs['training_args']['run_name']

        if tokenizer != None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.eval_args.tokenizer_path, 
                                                            pad_token='<pad>', 
                                                            padding_side='right')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_eval(self, current_model, step:int = 0, write_out: bool = False):
        #> setup
        tbl = PrettyTable()
        tbl.field_names = ["ds_name", "temp_name", "result"]
        self.model = current_model
        print(f"device: {self.accelerator.device}")

        #> load model
        print('Loading model...')
        print(f'Peft model id: {self.eval_args.peft_model_id}')
        multiple_choice_model = DecoderModel(self.eval_args.model_path, self.eval_args.peft_model_id, current_model)
        # causal_model = load_causal_model(self.eval_args.model_path, self.eval_args.peft_model_id, current_model)
        causal_model = current_model
        # Remove "Setting pad_token_id to eos_token_id" warning!
        # self.model.config.pad_token_id  = self.model.config.eos_token_id

        print(f'base model: {self.eval_args.model_path}')
        print("Note that if base model and peft model are 'None', the evaluation function is using the model under training!")

        multiple_choice_templates = TEMP_LIST['multiple_choice']
        generate_until_templates = TEMP_LIST['generate_until']
        eval_datasets = self.eval_args.datasets.split(',')
        task_type = self.eval_args.task_type.split(',')

        print(f"Eval datasets: {eval_datasets}")

        all_results = []
        samples = []
        if 'multiple_choice' in task_type:
            for ds_name, temp_list in multiple_choice_templates.items():
                if ds_name in eval_datasets:
                    for temp_name in temp_list:
                        res = self.run_multiple_choice_evaluation(ds_name, temp_name, multiple_choice_model)
                        all_results.append(res)
                        
                        # sample = self.generate_sample(ds_name, temp_name, causal_model)
                        # samples.append(sample)
                else:
                    continue

        if 'generate_until' in task_type:
            for ds_name, temp_list in generate_until_templates.items():
                if ds_name in eval_datasets:
                    for temp_name in temp_list:
                        res = self.run_generate_until_evaluation(ds_name, temp_name, causal_model)    
                        all_results.append(res)

                        # sample = self.generate_sample(ds_name, temp_name, causal_model)
                        # samples.append(sample)
                else:
                    continue

        with open(f'../evaluation_results/{self.run_name}.json', 'a+') as f:
            json.dump({f'Evaluation results at step {step}': all_results}, f)

        for res in all_results:
            tbl.add_row([res['ds_name'], res['temp_name'], res['result']])

        print(tbl)
        
        if write_out:
            with open(f'../evaluation/{self.run_name}_results.txt', 'w') as f:
                f.write(str(tbl))

        # print("#### Generated Samples ####")
        with open('../evaluation_results/samples.json', 'w+') as f:
            json.dump({f'Samples at step {step}': samples}, f)


        #samples = {f'Samples at step {step}': samples}
        #all_results = {'Evaluation results': all_results}
        # self.pretty_print(samples)
        # print("#### Generated Samples stored at 'sample.json'! ####")

        return all_results, samples
    
    def pretty_print(self, res):
        for item in res:
            for k, v in item.items():
                print(k,': ', v)
            print('\n')


    def generate_sample(self, ds_name, temp_name, model):
        #> load dataset
        self.val_set = FarsInstructEvalDataset(self.tokenizer, 
                                               max_len=self.eval_args.max_len, 
                                               instruction_template=self.eval_args.instruction_template,
                                               split=self.split,
                                               shots=self.shots)
        encoded_dataset = self.val_set.get_tokenized_data(ds_name=ds_name, temp_name=temp_name, multiple_choice=False)
        data_collator = DataCollatorWithPadding(self.tokenizer,
                                                return_tensors='pt')
        
        val_dataloader = DataLoader(encoded_dataset, collate_fn=data_collator, batch_size=1)
        val_dataloader, model = self.accelerator.prepare(val_dataloader, model)

        model.eval()
        batch = next(iter(val_dataloader))
        with torch.no_grad():
            generated_tokens = self.accelerator.unwrap_model(model).generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                 max_new_tokens=5
            )

            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            labels = batch["labels"]
            
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_inputs = self.tokenizer.batch_decode(generated_tokens[:, :batch['input_ids'].shape[1]], skip_special_tokens=True)
            decoded_preds = self.tokenizer.batch_decode(generated_tokens[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
            
            return {
                'ds_name': ds_name,
                'temp_name': temp_name,
                'tokens': [decoded_inputs, decoded_preds, decoded_labels]
                }

    def run_multiple_choice_evaluation(self, ds_name, temp_name, model):
        #> load dataset
        self.val_set = FarsInstructEvalDataset(self.tokenizer, 
                                               max_len=self.eval_args.max_len, 
                                               instruction_template=self.eval_args.instruction_template,
                                               split=self.split,
                                               shots=self.shots)
        encoded_dataset = self.val_set.get_tokenized_data(ds_name=ds_name, temp_name=temp_name, multiple_choice=True)
        data_collator = DataCollatorForMultipleChoice(self.tokenizer)
        val_dataloader = DataLoader(encoded_dataset, collate_fn=data_collator, batch_size=self.eval_args.batch_size)

        val_dataloader, model = self.accelerator.prepare(val_dataloader, model)
            
        #> start evaluation
        print(f"Start Evaluation on {ds_name}/{temp_name}...")
        model.eval()
        metric = evaluate.load("accuracy")
        for idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            with torch.no_grad():
                predictions = model(batch)

                metric.add_batch(
                predictions=predictions,
                references=batch["targets"])
                
        result = metric.compute()
        output_res = {
            'ds_name': ds_name, 
            'temp_name': temp_name, 
            'result': result
        }

        # print(output_res, '\n')

        return output_res

    # ----------------------------------------------------
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def run_generate_until_evaluation(self, ds_name, temp_name, model):
        #> load dataset
        self.val_set = FarsInstructEvalDataset(self.tokenizer, 
                                    max_len=self.eval_args.max_len, 
                                    instruction_template=self.eval_args.instruction_template,
                                    split=self.split,
                                    shots=self.shots)
        encoded_dataset = self.val_set.get_tokenized_data(ds_name=ds_name, temp_name=temp_name, multiple_choice=False)
        data_collator = DataCollatorWithPadding(self.tokenizer,
                                                return_tensors='pt')
        
        val_dataloader = DataLoader(encoded_dataset, collate_fn=data_collator, batch_size=self.eval_args.batch_size)
        val_dataloader, model = self.accelerator.prepare(val_dataloader, model)
        
        #> start evaluation
        print(f'Start evaluation on {ds_name}/{temp_name}...')
        model.eval()
        metric = evaluate.load('rouge')
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(model).generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = batch["labels"]
                
                generated_tokens, labels = self.accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

        result = metric.compute(use_stemmer=False)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        output_res = {
            'ds_name': ds_name, 
            'temp_name': temp_name, 
            'result': result
        }

        # print(output_res, '\n')

        return output_res


if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct Evaluation")
    parser.add_argument('--split', choices=['test', 'validation'], required=True)
    parser.add_argument('--write_out', action='store_true')
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')

    lm_eval = LMEvaluation(configs, None, args.split)
    lm_eval.run_eval(current_model=None, step=-1, write_out=args.write_out)
