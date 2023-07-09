import yaml
from dataclasses import dataclass

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
    vocab_size: int

@dataclass
class TrainingArgs:
    lr: float
    seed: int
    batch_size: int
    scheduler_step: int
    epochs: int
    pin_memory: bool
    buffer_size: int
    max_len: int


