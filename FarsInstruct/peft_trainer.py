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
        train_set = FarsInstructDataset(tokenizer, max_len=training_args.max_len, split='train', 
                                        stream=True, dataload_mode=args.dataload_mode, 
                                        dataset_path=data_args.dataset_path)
        train_set = train_set.get_tokenized_data(in_torch_format=False)
        train_set = train_set.shuffle(seed, buffer_size=training_args.buffer_size)

    else:
        train_set = FarsInstructDataset(tokenizer, max_len=training_args.max_len, split='train', 
                                        stream=False, dataload_mode=args.dataload_mode, 
                                        dataset_path=data_args.dataset_path)
        train_set = train_set.get_tokenized_data(in_torch_format=False)
    train_set = train_set.map(FarsInstructDataset.remove_mid_dim)
    
    #> start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    print('Start training...')
    trainer.train()

    trainer.save(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')



if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct")
    parser.add_argument('--dataload_mode', choices=['local', 'hub'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)
