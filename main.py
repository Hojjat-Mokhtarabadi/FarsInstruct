from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from transformers import Trainer, TrainingArguments

import torch
from torch.utils import data
from torch.optim import AdamW
from torch.cuda import amp
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

from dataset.fars_instruct_dataset import FarsInstructDataset
from model import load_model
from utils import *



def main(configs, args):
    #> setup
    data_args = configs['dataset_args']
    model_args = configs['model_args']
    training_args = configs['training_args']

    seed = training_args['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    # device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    device = 'cpu'

    #> load model
    print('Loading model...')
    model, tokenizer = load_model(model_args, args)
    model = model.to(device)

    #> load dataset
    train_set = FarsInstructDataset(tokenizer, max_len=1024, split='train')
    train_loader = data.DataLoader(train_set, batch_size=training_args['batch_size'], shuffle=True)
    
    #> load training misc
    print('Preparing training misc...')
    optimizer = AdamW(model.parameters(), training_args['lr'])
    training_steps = training_args['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear', 
        optimizer, 
        num_warmup_steps=training_args['scheduler_step'],
        num_training_steps=training_steps)
    scaler = amp.GradScaler()
    
    #> setup trainer
    progress_bar = tqdm(range(training_steps))
    epochs = 1
    for epoch in range(epochs):
        model.train()
        metrics = {'avg_loss' : [], 'acc' : []}
        for batch in train_loader:
            input_ids = batch[0].to(device)
            labels = batch[0].to(device)
            mask = batch[1].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float32):
                pred = model(input_ids, labels=labels, attention_mask=mask, token_type_ids=None, return_dict=True)
                loss = pred['loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()  
            scaler.step(optimizer)

            scaler.update()

            progress_bar.update(1)

        metrics['avg_loss'].append(loss.item())


        model.save_pretrained('./checkpoint')
        tokenizer.save_pretrained('./checkpoint')
    



if __name__ == "__main__":
    parse = ArgumentParser("Fars Insturct")
    args = parse.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)