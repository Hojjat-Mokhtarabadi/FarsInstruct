import numpy as np
import torch
from torch.utils import data
from torch.optim.optimizer import Optimizer as Optimizer
from transformers import Trainer, DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from argparse import ArgumentParser

from data_ops.fars_instruct_dataset import FarsInstructDataset
from modeling import load_pretaining_model
# from callbacks import LLMTensorboardCallback
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
    training_args = TrainingArgs(**configs['training_args'], )
    quantization_args = QuantizationArgs(**configs['quantization_args'])

    # config = ProjectConfiguration(project_dir=".", logging_dir=training_args.logging_dir)
    accelerator = Accelerator(cpu=False)
    # accelerator.init_trackers(
    #     project_name=training_args.run_name
    #     #config
    # )

    #if accelerator.is_main_process:
        # Initialise your wandb run, passing wandb parameters and any config information
    #    neptune_run = start_neptune_run(configs)

    #https://huggingface.co/docs/accelerate/en/usage_guides/tracking


    seed = training_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if accelerator.is_main_process:
        print(f"seed: {training_args.seed}")
        print('Loading model...')

    print(f"device: {accelerator.device}")
    #> load model
    # quantization_args = None
    model, tokenizer = load_pretaining_model(model_args.model_path, model_args.tokenizer_path, quantization_args)
    # tokenizer.pad_token = tokenizer.eos_token_id 
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(128001)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)
   
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

    lora_config = LoraConfig(
        r=quantization_args.lora_rank, 
        lora_alpha=quantization_args.lora_alpha, 
        lora_dropout=quantization_args.lora_dropout, 
        target_modules = target_modules,
        bias="none", 
        task_type="CAUSAL_LM"
    )

    if accelerator.is_main_process:
        print(f"Peft Model id: {model_args.peft_model}")
    if model_args.peft_model != None:
        model = PeftModel.from_pretrained(model, model_args.peft_model, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()

    if accelerator.is_main_process:
        print(f'base model: {model_args.model_path}')
        print_trainable_parameters(model)
        print('Preparing dataset...')

    #> load dataset
    if data_args.streaming:
        with accelerator.main_process_first():
            train_set = FarsInstructDataset(tokenizer, 
                                            max_len=training_args.max_len, 
                                            split='train', 
                                            stream=True, 
                                            dataload_mode=args.dataload_mode, 
                                            dataset_path=data_args.dataset_path, 
                                            instruction_template=training_args.instruction_template,
                                            datasets=training_args.datasets,
                                            shots=training_args.shots)
            train_set = train_set.get_tokenized_data(in_torch_format=False)
            train_set = train_set.shuffle(seed, buffer_size=training_args.buffer_size)

        train_loader = data.DataLoader(train_set, pin_memory=training_args.pin_memory,
                                       batch_size=training_args.per_device_train_batch_size)

    else:
        with accelerator.main_process_first():
            train_set = FarsInstructDataset(tokenizer, 
                                            max_len=training_args.max_len, 
                                            split='train', 
                                            stream=False, 
                                            dataload_mode=args.dataload_mode, 
                                            dataset_path=data_args.dataset_path, 
                                            instruction_template=training_args.instruction_template,
                                            datasets=training_args.datasets,
                                            shots=training_args.shots)
            train_set = train_set.get_tokenized_data(in_torch_format=True)
        
        random_sampler = data.RandomSampler(train_set, replacement=True, num_samples=training_args.max_steps if training_args.max_steps != -1 else len(train_set))
        train_loader = data.DataLoader(train_set, sampler=random_sampler, pin_memory=training_args.pin_memory,  
                                       batch_size=training_args.per_device_train_batch_size)


    
    accelerator.wait_for_everyone()
    #model, train_set = accelerator.prepare(model, train_set)
    if accelerator.is_main_process:
        print(f"Dataset length: {len(train_set)}")
        print("### Dataset sample: ###")
        print(tokenizer.batch_decode(next(iter(train_loader))['input_ids'])[0])

    response_template = " ### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        #tokenizer=tokenizer,
        data_collator=collator,
        #accelerator=accelerator,
    )
    
    # we instantiate the callback with the trainer object and the dataset we want to sample from
    # tensorboard_callback = LLMTensorboardCallback(trainer, configs, training_args.logging_dir, training_args.run_name )
    # trainer.add_callback(tensorboard_callback)

    if accelerator.is_main_process:
        print('Start training...')
    trainer.train(resume_from_checkpoint=model_args.peft_model)  
    # trainer.train()  


    # trainer.save(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')



if __name__ == "__main__":
    parser = ArgumentParser("Fars Insturct")
    parser.add_argument('--dataload_mode', choices=['local', 'hub'], required=True)
    args = parser.parse_args()
    configs = load_yml_file('confs.yaml')
    
    main(configs, args)


