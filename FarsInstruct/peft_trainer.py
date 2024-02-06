from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils import data
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from argparse import ArgumentParser
import numpy as np
import torch

from data_ops.fars_instruct_dataset import FarsInstructDataset
from modeling import load_pretaining_model
from utils import *
from callbacks import LLMSampleCB


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
    training_args = TrainingArgs(**configs['training_args'], )
    quantization_args = QuantizationArgs(**configs['quantization_args'])
    accelerator = Accelerator(cpu=False, log_with="wandb")

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



    model, train_set = accelerator.prepare(model, train_set)
    model.resize_token_embeddings(len(tokenizer))


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # we instantiate the W&B callback with the trainer object and the dataset we want to sample from
    wandb_callback = LLMSampleCB(trainer, train_set, configs, num_samples=10, max_new_tokens=256)
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


