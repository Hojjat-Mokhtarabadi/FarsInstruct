from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_metric
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import logging

from FarsInstruct.data_ops.fars_instruct_dataset import FarsInstructDataset
from FarsInstruct.data_ops.data_collator import DataCollatorForMultipleChoice
from FarsInstruct.modeling import load_model
from FarsInstruct import Phase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BASE_MODLE = "./checkpoints/hooshvare.zs-fs.digi-snapp.-1.bs16/"

def main():
    acc = Accelerator()
    metric = load_metric('accuracy')

    # load model
    model, tokenizer = load_model(phase=Phase.LABELING_EVALUATION, model_name_or_path=BASE_MODLE)

    
    # prepare dataset
    val_set = FarsInstructDataset(tokenizer=tokenizer, max_len=512, split='validation',
                        stream=False, dataload_mode='local', dataset_path=None)
    val_set = val_set.get_tokenized_data(in_torch_format=False)

    data_collator = DataCollatorForMultipleChoice(tokenizer)

    val_loader = DataLoader(val_set, collate_fn=data_collator, batch_size=1, pin_memory=True)

    # print(next(iter(val_loader)))
    # print(len(val_loader))

    val_loader, model = acc.prepare(val_loader, model)
    
    logger.info(f"device={acc.device}")

    model.eval()
    for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            predictions = model(batch)

        metric.add_batch(
            predictions=predictions,
            references=batch['targets']
        )

    eval_metric = metric.compute()
    print(eval_metric)        
    


if __name__ == "__main__":
    main()
