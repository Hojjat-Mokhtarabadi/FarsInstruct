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
import datasets

from transformers import DataCollatorWithPadding, AutoModelForCausalLM
from hazm import sent_tokenize

from prettytable import PrettyTable

from FarsInstruct.evaluation.data_collator import DataCollatorForMultipleChoice
from FarsInstruct.evaluation.eval_dataset import FarsInstructEvalDataset
from FarsInstruct.evaluation.model import DecoderModel, EncoderDecoderModel
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
                                                           use_fast=True,
                                                           add_bos_token=True,
                                                           padding_side='left'
                                                           )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_eval(self, current_model, step:int = 0, write_out: bool = False):
        #> setup
        tbl = PrettyTable(title=self.eval_args.run_name)
        tbl.field_names = ["ds_name", "temp_name", "result"]
        self.model = current_model
        print(f"device: {self.accelerator.device}")
        
        multiple_choice_templates = TEMP_LIST['multiple_choice']
        generate_until_templates = TEMP_LIST['generate_until']
        eval_datasets = self.eval_args.datasets.split(',')
        task_type = self.eval_args.task_type.split(',')

        #> load model
        print('Loading model...')
        print(f'Peft model id: {self.eval_args.peft_model_id}')

        if 'multiple_choice' in task_type:
            if self.eval_args.model_type == 'causal':
                multiple_choice_model = DecoderModel(self.eval_args.model_path, self.eval_args.peft_model_id, current_model)
            elif self.eval_args.model_type == 'seq2seq':
                multiple_choice_model = EncoderDecoderModel(self.eval_args.model_path, self.eval_args.peft_model_id, current_model)
            else:
                raise Exception("Invalid model type!")
        else:
            causal_model = AutoModelForCausalLM.from_pretrained(self.eval_args.model_path,
                                                #quantization_config=quantization_config,
                                                torch_dtype=torch.float16,
                                                device_map="auto")
        # Remove "Setting pad_token_id to eos_token_id" warning!
        # self.model.config.pad_token_id  = self.model.config.eos_token_id

        print(f'base model: {self.eval_args.model_path}')
        print("Note that if base model and peft model are 'None', the evaluation function is using the model under training!")

        print(f"Eval datasets: {eval_datasets}")

        all_results = []
        samples = {}
        set_lst = []
        if 'multiple_choice' in task_type:
            for ds in eval_datasets:
                temp_list = multiple_choice_templates[ds]
                for temp_name in temp_list:
                    res = self.run_multiple_choice_evaluation(ds, temp_name, multiple_choice_model)
                    all_results.append(res)
        
        if 'generate_until' in task_type:
            for ds in eval_datasets:
                all_scores = []
                temp_list = []
                try:
                    temp_list1 = generate_until_templates[ds]
                    for temp_name in temp_list1:
                        res, scores, dec_preds, dec_labels = self.run_generate_until_evaluation(ds, temp_name, causal_model)    
                        all_results.append(res)
                        all_scores.append(scores)
                        temp_list += temp_list1
                        samples[temp_name] = [dec_preds, dec_labels]
                except Exception as e:
                    print(f"{e}")
                    
                try:
                    temp_list2 = multiple_choice_templates[ds]
                    for temp_name in temp_list2:
                        res, scores, dec_preds, dec_labels = self.run_generate_until_evaluation(ds, temp_name, causal_model)    
                        all_results.append(res)
                        all_scores.append(scores)
                        temp_list += temp_list2
                        samples[temp_name] = [dec_preds, dec_labels]
                except Exception as e:
                    print(f"{e}")
                    
                
                set_lst.append(self.create_set(ds, temp_list, all_scores))


        # with open(f'../evaluation_results/{self.run_name}.json', 'a+') as f:
        #     json.dump({f'Evaluation results at step {step}': all_results}, f)


        # Combine tables into one
        combined_table = PrettyTable(title=self.eval_args.run_name)
        combined_table.field_names = ["Section", "Score", "Type", "Precision", "Recall", "F-Measure"]
        
        # Add sets to combined table
        for idx, ds in enumerate(eval_datasets):
            # print(set_lst)
            self.add_set_to_combined(ds, set_lst[idx], combined_table)

        print(combined_table)        
        if write_out:
            with open(f'../evaluation_results/{self.eval_args.run_name}_results.txt', 'w') as f:
                f.write(str(combined_table))
            with open(f'../evaluation_results/{self.eval_args.run_name}_samples.json', 'w+') as f:
                json.dump(samples, f)


        # # print("#### Generated Samples ####")
        # with open('../evaluation_results/samples.json', 'w+') as f:
        #     json.dump({f'Samples at step {step}': samples}, f)


        # samples = {f'Samples at step {step}': samples}
        #all_results = {'Evaluation results': all_results}
        # self.pretty_print(samples)
        # print("#### Generated Samples stored at 'sample.json'! ####")

        return all_results, samples

    
    def create_table(self, name, scores):
        table = PrettyTable()
        table.field_names = ["Score", "Type", "Precision", "Recall", "F-Measure"]
        table.title = name

        for i, score_name in enumerate(scores):
            score = dict(scores[score_name]._asdict())
            for score_type in score:
                values = score[score_type]
                if score_name == "rougeL":
                    table.add_row([f'{score_name}', f'{score_type}', f'{values.precision * 100:0.2f} ', f'{values.recall * 100:0.2f}', f'{values.fmeasure * 100:0.2f}'])
        
        return table

    def create_set(self, set_name, table_names, scores):
        tables = []
        for name, score in zip(table_names, scores):
            tables.append(self.create_table(name, score))
        return tables
    
    def add_set_to_combined(self, set_name, tables, combined_table):
        combined_table.add_row([set_name, "", "", "", "", ""])
        for i, table in enumerate(tables):
            section_name = f"{set_name} - {table.title}"
            self.add_table_to_combined(table, section_name, combined_table)
    
    def add_table_to_combined(self, table, section_name, combined_table):
        combined_table.add_row([section_name, "", "", "", "", ""])
        for row in table.rows:
            combined_table.add_row(["", *row])
    
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
                max_new_tokens=20
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

    # -------------------------------------------------------
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
        metric = datasets.load_metric('rouge')
        dec_preds = []
        dec_labels = []
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(model).generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=64,
                    top_k=10,
                    eos_token_id=128001
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
                    
                decoded_preds = self.tokenizer.batch_decode(generated_tokens[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

            dec_preds = decoded_preds
            dec_labels = decoded_labels

        scores = metric.compute(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], 
            use_stemmer=False, 
            lang='fa'
        )
        # result = metric.compute(use_stemmer=False)
        # result = {k: round(v * 100, 4) for k, v in result.items()}

        output_res = {
            'ds_name': ds_name, 
            'temp_name': temp_name
        }

        # print(output_res, '\n')

        return output_res, scores, dec_preds, dec_labels


if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct Evaluation")
    parser.add_argument('--split', choices=['test', 'validation'], required=True)
    parser.add_argument('--write_out', action='store_true')
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')

    lm_eval = LMEvaluation(configs, None, args.split)
    all_result, sample = lm_eval.run_eval(current_model=None, step=-1, write_out=args.write_out)
