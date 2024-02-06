import numpy as np
from accelerate import Accelerator
import torch
from torch.utils import data
from argparse import ArgumentParser
from transformers import Trainer, DataCollatorForLanguageModeling
import warnings


from data_ops.fars_instruct_dataset import FarsInstructDataset
from callbacks import LLMSampleCB
from modeling import load_pretaining_model
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
    accelerator = Accelerator(cpu=False,log_with="wandb")
    print(f"seed: {training_args.seed}")
    print(f"device: {accelerator.device}")

    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name="FarsInstruct", 
        #config={"dropout": 0.1, "learning_rate": 1e-2}
        init_kwargs={"wandb": {"entity": "farsinstruct"}}
        )

    #> load model
    print('Loading model...')
    model, tokenizer = load_pretaining_model(model_args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    print(f'base model: {model_args.model_path}')
    print('number of parameters={:,}'.format(model.num_parameters()))

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
                                        datasets=training_args.datasets
                                        )
        train_set = train_set.get_tokenized_data(in_torch_format=True)
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
        

    model, train_set = accelerator.prepare(model, train_set)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # we instantiate the W&B callback with the trainer object and the dataset we want to sample from
    wandb_callback = LLMSampleCB(trainer, configs)
    trainer.add_callback(wandb_callback)

    print('Start training...')
    trainer.train()  

    # trainer.save(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')
  



if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct")
    parser.add_argument('--dataload_mode', choices=['local', 'hub'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)
