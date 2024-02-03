from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils import data
from transformers import get_scheduler
from torch.optim import AdamW
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm

from data_ops.fars_instruct_dataset import FarsInstructDataset
from FarsInstruct.evaluation.run_eval import run_eval
from modeling import load_pretaining_model
from utils import *
    

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(configs, args):
    #> setup
    data_args = DatasetArgs(**configs['dataset_args'])
    model_args = ModelArgs(**configs['model_args'])
    training_args = TrainingArgs(**configs['training_args'])
    quantization_args = QuantizationArgs(**configs['quantization_args'])

    seed = training_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator = Accelerator(cpu=False)
    print(f"seed: {training_args.seed}")
    print(f"device: {accelerator.device}")

     #> load model
    print('Loading model...')
    model, tokenizer = load_pretaining_model(model_args.model_path, quantization_args)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)
 
    lora_config = LoraConfig(
        r=quantization_args.lora_rank, 
        lora_alpha=quantization_args.lora_alpha, 
        lora_dropout=quantization_args.lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print(f'base model: {model_args.model_path}')
    print_trainable_parameters(model)

    #> load dataset
    print('Preparing dataset...')
    if data_args.streaming:
        train_set = FarsInstructDataset(tokenizer, 
                                        max_len=training_args.max_len, 
                                        split='train', 
                                        stream=True, 
                                        dataload_mode=args.dataload_mode, 
                                        dataset_path=data_args.dataset_path, 
                                        instruction_template=training_args.instruction_template,
                                        datasets=training_args.datasets)
        train_set = train_set.get_tokenized_data(in_torch_format=False)
        train_set = train_set.shuffle(seed, buffer_size=training_args.buffer_size)

        train_loader = data.DataLoader(train_set, pin_memory=training_args.pin_memory,
                                       batch_size=training_args.per_device_train_batch_size)

    else:
        train_set = FarsInstructDataset(tokenizer, 
                                        max_len=training_args.max_len, 
                                        split='train', 
                                        stream=False, 
                                        dataload_mode=args.dataload_mode, 
                                        dataset_path=data_args.dataset_path, 
                                        instruction_template=training_args.instruction_template,
                                        datasets=training_args.datasets)
        train_set = train_set.get_tokenized_data(in_torch_format=True)
        
        random_sampler = data.RandomSampler(train_set, replacement=True, num_samples=training_args.max_steps if training_args.max_steps != -1 else len(train_set))
        train_loader = data.DataLoader(train_set, sampler=random_sampler, pin_memory=training_args.pin_memory,  
                                       batch_size=training_args.per_device_train_batch_size)


    
    #> load training misc
    print('Preparing training misc...')
    optimizer = AdamW(model.parameters(), training_args.learning_rate)
    num_training_steps = training_args.num_train_epochs * len(train_loader) if training_args.max_steps == -1 else training_args.max_steps
    print(f'Num training steps: {num_training_steps}')

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
    
    #> setup trainer
    print("Start training...")
    #progress_bar = tqdm(range(training_steps))
    epochs = training_args.num_train_epochs

    for epoch in range(epochs):
        if data_args.streaming:
            train_set.set_epoch(epoch)
        model.train()
        metrics = {'avg_loss' : [], 'acc' : []}
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_ids = batch['input_ids']
            labels = batch['input_ids']
            mask = batch['attention_mask']

            input_ids = input_ids.squeeze(1)
            mask = mask.squeeze(1)
            labels = labels.squeeze(1)
            pred = model(input_ids=input_ids, labels=labels, attention_mask=mask, return_dict=True)
            loss = pred['loss']

            accelerator.backward(loss)  
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

            metrics['avg_loss'].append(loss.item())

            if idx % training_args.logging_steps == 0:
                print("#### Running Evaluation... ####")
                run_eval(configs, split='validation')
                print(f"Avg loss: {sum(metrics['avg_loss']) / len(metrics['avg_loss'])}", '\n')
            # Log to wandb by calling `accelerator.log`, `step` is optional
            accelerator.log({"avg_loss": sum(metrics['avg_loss'])/ len(metrics['avg_loss'])})#, step=global_step)

            if idx+1 % training_args.save_steps == 0:
                model.save_pretrained(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')
                tokenizer.save_pretrained(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')

    # Make sure that the wandb tracker finishes correctly
    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct")
    parser.add_argument('--dataload_mode', choices=['local', 'hub'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)
