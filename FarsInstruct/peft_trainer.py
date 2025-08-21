import numpy as np
import torch
from torch.utils import data
from torch.optim.optimizer import Optimizer as Optimizer
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from argparse import ArgumentParser

from data_ops.collators import DataCollatorForCompletionLM
from data_ops.farsinstruct.farsinstruct_dataset import FarsInstructDataset
from data_ops.sni.sni_dataset import SNIDataset
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
    
    # Initialize accelerator
    accelerator = Accelerator()

    # config = ProjectConfiguration(project_dir=".", logging_dir=training_args.logging_dir)
    # accelerator = Accelerator(cpu=False)
    seed = training_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU reproducibility
    
    # Main process check for prints/logging
    # is_main_process = training_args.local_rank in [-1, 0]

    if accelerator.is_main_process:
        # Setup logging
        transformers.utils.logging.set_verbosity_info()

        # Log on each process the small summary
        print(f"Process rank: {accelerator.process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed training: {accelerator.distributed_type}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"seed: {training_args.seed}")
        print('Loading model...')

    # print(f"device: {accelerator.device}")
    #> load model
    # quantization_args = None
    model, tokenizer = load_pretaining_model(model_args.model_path, model_args.tokenizer_path, quantization_args)
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.gradient_checkpointing_enable()
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)
   
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']

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

    # model.resize_token_embeddings(len(tokenizer))
    # model.enable_input_require_grads()

    if accelerator.is_main_process:
        print(f'base model: {model_args.model_path}')
        print_trainable_parameters(model)
        print('Preparing dataset...')

    #> load dataset
    # data collator for sft training
    sft_collator = DataCollatorForCompletionLM(tokenizer=tokenizer, model_family=model_args.model_family)
    
    with accelerator.main_process_first():
        if data_args.dataset_family == "farsinstruct":
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
            random_sampler = data.RandomSampler(train_set, 
                                                replacement=True, 
                                                num_samples=training_args.max_steps if training_args.max_steps != -1 else len(train_set))
            train_loader = data.DataLoader(train_set, 
                                           sampler=random_sampler, 
                                           pin_memory=training_args.pin_memory,  
                                           batch_size=training_args.per_device_train_batch_size,
                                           collate_fn=sft_collator)

        elif data_args.dataset_family == "sni":
            train_set = SNIDataset(tokenizer,
                                   max_len=training_args.max_len,
                                   max_task_examples=data_args.max_examples,
                                   lang=data_args.language,
                                   categories=data_args.categories,
                                   model_family=model_args.model_family)
            train_set = train_set.get_tokenized_data(in_torch_format=True).shuffle(seed=training_args.seed)
            train_loader = data.DataLoader(train_set,
                                           batch_size=training_args.per_device_train_batch_size, 
                                           collate_fn=sft_collator)
    
            
    accelerator.wait_for_everyone()
    model, train_set = accelerator.prepare(model, train_set)
    if accelerator.is_main_process:
        print(f"Dataset length: {len(train_set)}")
        print("### Dataset sample: ###")
        print("Input: ", tokenizer.batch_decode(next(iter(train_loader))['input_ids'])[0])
        print("Label: ", next(iter(train_loader))["labels"][0])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=sft_collator,
        #accelerator=accelerator,
    )
    
    if hasattr(model, 'is_parallelizable'):
        model.is_parallelizable = True
    if hasattr(model, 'model_parallel'):
        model.model_parallel = True
    
    # we instantiate the callback with the trainer object and the dataset we want to sample from
    # tensorboard_callback = LLMTensorboardCallback(trainer, configs, training_args.logging_dir, training_args.run_name )
    # trainer.add_callback(tensorboard_callback)

    if accelerator.is_main_process:
        print('Start training...')
    trainer.train(resume_from_checkpoint=model_args.peft_model)  
    # trainer.train()  

    # trainer.save(f'./checkpoints/{training_args.desc}.{training_args.max_steps}.bs{training_args.per_device_train_batch_size}')


if __name__ == "__main__":
    parser = ArgumentParser("Global Co-Cola")
    args = parser.parse_args()
    configs = load_yml_file('configs/sni_confs.yaml')
    
    main(configs, args)


