import yaml
from dataclasses import dataclass
from transformers import TrainingArguments

def load_yml_file(pth):
    with open(pth, 'r') as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as y:
            print(y)

    return configs

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




