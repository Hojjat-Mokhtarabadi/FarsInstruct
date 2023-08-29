from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_metric
from FarsInstruct.data_ops.fars_instruct_dataset import FarsInstructDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

BASE_MODLE = "HooshvareLab/gpt2-fa"

def load_model():
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'emp_token': '<emp>'
    }
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODLE, **special_tokens)
    config = AutoConfig.from_pretrained(BASE_MODLE)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODLE, config=config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def main():
    metric = load_metric('accuracy')

    # load model
    model, tokenizer = load_model()
    
    # prepare dataset
    val_set = FarsInstructDataset(tokenizer=tokenizer, max_len=256, split='validation',
                        stream=False, dataload_mode='local', dataset_path=None)
    val_set = val_set.get_tokenized_data(in_torch_format=True)
    val_loader = DataLoader(val_set, batch_size=5, pin_memory=True)

    #print(next(iter(val_loader)))
    print(len(val_loader))

    model.eval()
    for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        print(idx)
        with torch.no_grad():
            predictions = model(batch['input_ids'], attention_mask=batch['attention_mask'])

        print(metric.compute(references=[0, 1], predictions=[1, 1]))

        print(batch['targets'])

        metric.add_batch(
            predictions=predictions,
            references=batch['targets']
        )

    eval_metric = metric.compute()
    print(eval_metric)        
    


    

   


if __name__ == "__main__":
    main()
