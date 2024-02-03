from torch.utils.data import DataLoader    
import torch
from tqdm import tqdm
import argparse
from argparse import ArgumentParser
import warnings
from transformers import AutoTokenizer
import evaluate
from accelerate import Accelerator
import numpy as np
from transformers import DataCollatorWithPadding
from hazm import sent_tokenize
import json
from prettytable import PrettyTable

from FarsInstruct.evaluation.data_collator import DataCollatorForMultipleChoice
from FarsInstruct.evaluation.eval_dataset import FarsInstructEvalDataset
from FarsInstruct.evaluation.model import DecoderModel, load_causal_model
from FarsInstruct.evaluation.temp_list import TEMP_LIST

from FarsInstruct.utils import EvaluationArgs, load_yml_file, DatasetArgs

#! ignore sourceTensor.clone().detach() warning
warnings.filterwarnings("ignore", category=UserWarning)

def run_multiple_choice_evaluation(eval_args, data_args, ds_name, temp_name, tokenizer, model, accelerator, split):
    #> load dataset
    val_set = FarsInstructEvalDataset(tokenizer, 
                                      max_len=eval_args.max_len, 
                                      instruction_template=eval_args.instruction_template,
                                      split=split)
    
    encoded_dataset = val_set.get_tokenized_data(ds_name=ds_name, temp_name=temp_name, multiple_choice=True)
    data_collator = DataCollatorForMultipleChoice(tokenizer)
    val_dataloader = DataLoader(encoded_dataset, collate_fn=data_collator, batch_size=eval_args.batch_size)

    val_dataloader, model = accelerator.prepare(val_dataloader, model)
        
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
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def run_generate_until_evaluation(eval_args, data_args, ds_name, temp_name, tokenizer, model, accelerator, split):
    #> load dataset
    val_set = FarsInstructEvalDataset(tokenizer, 
                                      max_len=eval_args.max_len, 
                                      instruction_template=eval_args.instruction_template, 
                                      split=split)
    
    encoded_dataset = val_set.get_tokenized_data(ds_name=ds_name, temp_name=temp_name, multiple_choice=False)
    data_collator = DataCollatorWithPadding(tokenizer,
                                            return_tensors='pt')
    
    val_dataloader = DataLoader(encoded_dataset, collate_fn=data_collator, batch_size=eval_args.batch_size)
    val_dataloader, model = accelerator.prepare(val_dataloader, model)
    
    #> start evaluation
    print(f'Start evaluation on {ds_name}/{temp_name}...')
    model.eval()
    metric = evaluate.load('rouge')
    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            
            generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
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


def run_eval(configs, split):
    #> setup
    accelerator = Accelerator(cpu=False)
    eval_args = EvaluationArgs(**configs['evaluation_args'])
    data_args = DatasetArgs(**configs['dataset_args'])
    tbl = PrettyTable()
    tbl.field_names = ["ds_name", "temp_name", "result"]

    print(f"device: {accelerator.device}")

    #> load model
    print('Loading model...')
    print(f'Peft model id: {eval_args.peft_model_id}')
    multiple_choice_model = DecoderModel(eval_args.model_path, eval_args.peft_model_id)
    causal_model = load_causal_model(eval_args.model_path, eval_args.peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_path, 
                                              pad_token='<pad>', 
                                              padding_side='right')
    print(f'base model: {eval_args.model_path}')

    multiple_choice_templates = TEMP_LIST['multiple_choice']
    generate_until_templates = TEMP_LIST['generate_until']
    eval_datasets = eval_args.datasets.split(',')
    task_type = eval_args.task_type.split(',')

    print(f"Eval datasets: {eval_datasets}")

    all_results = []
    if 'multiple_choice' in task_type:
        for ds_name, temp_list in multiple_choice_templates.items():
            if ds_name in eval_datasets:
                for temp_name in temp_list:
                    res = run_multiple_choice_evaluation(eval_args, data_args, ds_name, temp_name, 
                                                         tokenizer, multiple_choice_model, accelerator, split)
                    all_results.append(res)
            else:
                continue

    if 'generation' in task_type:
        for ds_name, temp_list in generate_until_templates.items():
            if ds_name in eval_datasets:
                for temp_name in temp_list:
                    res = run_generate_until_evaluation(eval_args, data_args, ds_name, temp_name, 
                                                        tokenizer, causal_model, accelerator, split)    
                    all_results.append(res)
            else:
                continue

    with open('../evaluation_results/results.json', 'w') as f:
        json.dump({'Evaluation results': all_results}, f)

    for res in all_results:
        tbl.add_row([res['ds_name'], res['temp_name'], res['result']])

    print(tbl)


if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct Evaluation")
    parser.add_argument('--split', choices=['test', 'validation'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')

    
    run_eval(configs, args.split)
