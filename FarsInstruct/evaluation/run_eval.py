from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import metric
from FarsInstruct.data_ops.fars_instruct_dataset import FarsInstructDataset

BASE_MODLE = "HooshvareLab/gpt2-fa"

def load_model():
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODLE, **special_tokens)
    config = AutoConfig.from_pretrained(BASE_MODLE)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODLE, config=config)

    return model, tokenizer


def main():
    # load model
    model, tokenizer = load_model()

    # prepare dataset
    val_set = FarsInstructDataset(tokenizer=tokenizer, max_len=256, split='validation',
                        stream=False, dataload_mode='local', dataset_path=None)
    val_set = val_set.get_tokenized_data(in_torch_format=False)

   


if __name__ == "__main__":
    main()
