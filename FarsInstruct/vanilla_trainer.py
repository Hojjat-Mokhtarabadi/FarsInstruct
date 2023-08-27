import numpy as np
from accelerate import Accelerator
import torch
from torch.utils import data
from torch.optim import AdamW
from torch.cuda import amp
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import get_scheduler
import warnings
import random

from FarsInstruct.data_ops.fars_instruct_dataset import FarsInstructDataset
from model import load_model
from utils import *

#! ignore sourceTensor.clone().detach() warning
warnings.filterwarnings("ignore", category=UserWarning)

def main(configs, args):
    #> setup
    data_args = DatasetArgs(**configs['dataset_args'])
    model_args = ModelArgs(**configs['model_args'])
    training_args = TrainingArgs(**configs['training_args'])

    seed = training_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator = Accelerator(cpu=False)
    print(f"seed: {training_args.seed}")
    print(f"device: {accelerator.device}")

    #> load model
    print('Loading model...')
    model, tokenizer = load_model(model_args)
    model.resize_token_embeddings(len(tokenizer))

    print(f'base model: {model_args.model_path}')
    print('number of parameters={:,}'.format(model.num_parameters()))

    #> load dataset
    print('Preparing dataset...')
    if data_args.streaming:
        train_set = FarsInstructDataset(tokenizer, max_len=training_args.max_len, split='train', 
                                        stream=True, dataload_mode=args.dataload_mode, 
                                        dataset_path=data_args.dataset_path)
        train_set = train_set.get_tokenized_data(in_torch_format=True)
        train_set = train_set.shuffle(seed, buffer_size=training_args.buffer_size)

        train_loader = data.DataLoader(train_set, pin_memory=training_args.pin_memory,
                                       batch_size=training_args.per_device_train_batch_size)

    else:
        train_set = FarsInstructDataset(tokenizer, max_len=training_args.max_len, split='train', 
                                        stream=False, dataload_mode=args.dataload_mode, 
                                        dataset_path=data_args.dataset_path)
        train_set = train_set.get_tokenized_data(in_torch_format=True)
        #htrain_set = train_set.get_tokenized_data(in_torch_format=False)

        #subset_idx = list(range(training_args.max_steps))
        #subset_idx.shuffle(
        #subset_train_set = data.Subset(train_set, subset_idx)
        print(training_args.max_steps)
        random_sampler = data.RandomSampler(train_set, replacement=True, num_samples=training_args.max_steps if training_args.max_steps != -1 else len(train_set))
        train_loader = data.DataLoader(train_set, sampler=random_sampler, pin_memory=training_args.pin_memory,  
                                       batch_size=training_args.per_device_train_batch_size)

    #> load training misc
    print('Preparing training misc...')
    optimizer = AdamW(model.parameters(), training_args.learning_rate)
    training_steps = training_args.max_steps
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
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

            pred = model(input_ids, labels=labels, attention_mask=mask, token_type_ids=None, return_dict=True)
            loss = pred['loss']

            optimizer.zero_grad()
            accelerator.backward(loss)  
            optimizer.step()

     #       progress_bar.update(1)

        metrics['avg_loss'].append(loss.item())


        model.save_pretrained(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')
        tokenizer.save_pretrained(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')
    


if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct")
    parser.add_argument('--dataload_mode', choices=['local', 'hub'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)
