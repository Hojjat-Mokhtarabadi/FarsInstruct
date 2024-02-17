import yaml
from dataclasses import dataclass
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


def start_neptune_run(input_config, tags=[]):
    token = os.getenv("NEPTUNE_API_TOKEN")
    project = os.getenv("NEPTUNE_PROJECT")
    
    run = neptune.init_run(
        project=project,
        api_token=token,
    )
    run["host_machine"] = gather_host_params()

    run["data_args"] = input_config['dataset_args']
    run["model_args"] = input_config['model_args']
    run["training_args"] = input_config['training_args']
    run["quantization_args"] = input_config['quantization_args']

    tags.append(input_config['model_args']['model_path'].split('/')[-1])

    tags.append(f"trainig_dataset={input_config['training_args']['datasets']}")
    tags.append(f"evaluation_dataset={input_config['evaluation_args']['datasets']}")
    
    tags.append(input_config['training_args']['run_name'])
    tags.append(f"shots={input_config['training_args']['shots']}")
    tags.append(f"epochs={input_config['training_args']['num_train_epochs']}")
    tags.append(f"max_steps={input_config['training_args']['max_steps']}")

    
    if tags:
        run["sys/tags"].add(tags)
        
    return run

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
    dataset_path: str
    streaming: bool

@dataclass
class ModelArgs:
    model_path: str
    tokenizer_path: str
    vocab_size: int

class TrainingArgs(TrainingArguments):
  def __init__(self, datasets, instruction_template, shots, buffer_size, max_len, pin_memory, **kwargs):
    super().__init__(**kwargs)
    self.buffer_size: int = buffer_size
    self.max_len: int = max_len
    self.pin_memory: bool = pin_memory
    self.datasets: str = datasets
    self.instruction_template: str = instruction_template
    self.shots: int = shots

@dataclass
class EvaluationArgs:
    model_path: str
    tokenizer_path: str
    peft_model_id: str 
    batch_size: int
    max_len: int
    datasets: str
    instruction_template: str
    shots: int
    task_type: str

@dataclass
class QuantizationArgs:
    load_in_4bit: bool
    double_quant: bool
    quant_type: str
    lora_rank: int 
    lora_alpha: int 
    lora_dropout: float 




