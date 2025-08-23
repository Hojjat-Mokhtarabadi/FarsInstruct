import json
from pathlib import Path
import yaml
from typing import List
from dataclasses import dataclass, field
from transformers import TrainingArguments
import os
#import neptune
import platform


def save_training_args(data_args, model_args, train_args, quant_args, path):
    data_args = vars(data_args)
    model_args = vars(model_args)
    train_args = vars(train_args)
    quant_args = vars(quant_args)
    all_args = data_args | model_args | train_args | quant_args
    
    my_args = ["run_name", "dataset_family", "language", "categories", "max_examples", "model_family", "model_path", "max_len", "output_dir", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "num_train_epochs", "save_steps", "seed", "bf16", "fp16", "lora_rank", "lora_alpha", "lora_dropout", "load_in_8bit", "load_in_4bit", "quant_type"]
    ret_args = {}
    
    for i in my_args:
        for k, v in all_args.items():
            if k == i:
                ret_args[k] = v
    
    with open(Path(path) / "training_args.json" , "w") as file:
        json.dump(ret_args, file, indent=4)

def load_yml_file(pth):
    with open(pth, 'r') as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as y:
            print(y)

    return configs

def gather_host_params():
    host_params = {
        "system": platform.system(),
        "machine": platform.machine(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "uname": " ".join(platform.uname()),
        "cpu": platform.processor(),
    }
    return host_params

@dataclass
class DatasetArgs:
    dataset_family: str
    dataset_path: str
    max_examples: int
    language: str
    categories: List[str]

@dataclass
class ModelArgs:
    model_family: str
    model_path: str
    tokenizer_path: str
    peft_module_path: str = None

class TrainingArgs(TrainingArguments):
    def __init__(self, buffer_size, max_len, pin_memory, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size: int = buffer_size
        self.max_len: int = max_len
        self.pin_memory: bool = pin_memory
    
@dataclass
class EvaluationArgs:
    run_name: str
    model_path: str
    tokenizer_path: str
    model_type: str
    peft_model_id: str 
    batch_size: int
    max_len: int
    datasets: str
    instruction_template: str
    shots: int
    task_type: str

@dataclass
class QuantizationArgs:
    load_in_8bit: bool
    load_in_4bit: bool
    double_quant: bool
    quant_type: str
    lora_rank: int 
    lora_alpha: int 
    lora_dropout: float 




