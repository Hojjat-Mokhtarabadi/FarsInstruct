from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from transformers import BitsAndBytesConfig
import torch
from torch import nn
from argparse import ArgumentParser
from models.decoder_model import DecoderModel
from FarsInstruct import Phase


def load_model(phase: str, model_name_or_path, quantization_args=None):
    if quantization_args:
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_args.load_in_4bit,
                bnb_4bit_use_double_quant=quantization_args.double_quant,
                bnb_4bit_quant_type=quantization_args.quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              use_fast=True, 
                                              bos_token='<s>', 
                                              eos_token='</s>', 
                                              pad_token='<pad>')
    config = AutoConfig.from_pretrained(model_name_or_path)

    if phase == Phase.INSTRUCTION_TUNING:
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, quantization_config=bnb_config if quantization_args else None, 
                                                config=config, device_map="auto")
    elif phase == Phase.LABELING_EVALUATION:
        model = DecoderModel(model_name_or_path, config, device_map="auto")

    return model, tokenizer


if __name__ == "__main__":
    parser = ArgumentParser('Model test')
    args = parser.parse_args()

    sent = 'فردا یک قرار کاری با علی دارم'
    #model, tokenizer = load_model(args)




