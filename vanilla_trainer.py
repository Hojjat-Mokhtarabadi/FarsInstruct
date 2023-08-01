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

from dataset.fars_instruct_dataset import FarsInstructDataset
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
        
        train_loader = data.DataLoader(train_set, pin_memory=training_args.pin_memory,  
                                       batch_size=training_args.per_device_train_batch_size, shuffle=True)

    #> load training misc
    print('Preparing training misc...')
    optimizer = AdamW(model.parameters(), training_args.lr)
    dataset_len = 987841
    training_steps = training_args.epochs * dataset_len
    scheduler = get_scheduler(
        'linear', 
        optimizer, 
        num_warmup_steps=training_args.scheduler_step,
        num_training_steps=training_steps)
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    
    #> setup trainer
    print("Start training...")
    progress_bar = tqdm(range(training_steps))
    epochs = training_args.epochs
    for epoch in range(epochs):
        if data_args.streaming:
            train_set.set_epoch(epoch)
        model.train()
        metrics = {'avg_loss' : [], 'acc' : []}
        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids']
            labels = batch['input_ids']
            mask = batch['attention_mask']

            pred = model(input_ids, labels=labels, attention_mask=mask, token_type_ids=None, return_dict=True)
            loss = pred['loss']

            optimizer.zero_grad()
            accelerator.backward(loss)  
            optimizer.step()
            scheduler.step()

            progress_bar.update(1)

        metrics['avg_loss'].append(loss.item())


        model.save_pretrained('./checkpoint')
        tokenizer.save_pretrained('./checkpoint')
    


if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct")
    parser.add_argument('--dataload_mode', choices=['local', 'hub'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)