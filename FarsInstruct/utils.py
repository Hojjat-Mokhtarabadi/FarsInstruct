import yaml
from typing import List
from dataclasses import dataclass, field
from transformers import TrainingArguments
import os
#import neptune
import platform

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
    language: str
    categories: List[str]

@dataclass
class ModelArgs:
    model_family: str
    model_path: str
    tokenizer_path: str
    peft_model: str = None

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
    double_quant: bool
    quant_type: str
    lora_rank: int 
    lora_alpha: int 
    lora_dropout: float 




