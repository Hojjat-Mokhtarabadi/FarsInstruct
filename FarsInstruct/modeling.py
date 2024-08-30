import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from argparse import ArgumentParser



def load_pretaining_model(model_name_or_path, tokenizer_path, quantization_args=None):
    if quantization_args:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quantization_args.double_quant,
            bnb_4bit_quant_type=quantization_args.quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
  
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=True,
                                              padding_side='left',
                                              add_bos_token = True,
                                              add_eos_token = True)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 quantization_config=bnb_config if quantization_args else None,
                                                 config=config)

    return model, tokenizer


if __name__ == "__main__":
    parser = ArgumentParser('Model test')
    args = parser.parse_args()

    sent = 'فردا یک قرار کاری با علی دارم'
    #model, tokenizer = load_model(args)
