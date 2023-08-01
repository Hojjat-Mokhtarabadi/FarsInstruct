from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from transformers import BitsAndBytesConfig
import torch
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_path')

    return parser

def load_model(configs, quantization_args=None):
    if quantization_args:
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_args.load_in_4bit,
                bnb_4bit_use_double_quant=quantization_args.double_quant,
                bnb_4bit_quant_type=quantization_args.quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
    model_path = configs.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=True, 
                                              bos_token='<s>', 
                                              eos_token='</s>', 
                                              pad_token='<pad>')
    config = AutoConfig.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, quantization_config=bnb_config if quantization_args else None, 
                                            config=config, device_map={"":0})

    return model, tokenizer

def do_inference(model, tokenizer, input):
    input_map = tokenizer(input, return_tensors='pt')
    # generated = torch.tensor(tokenizer.encode(input)).unsqueeze(0)

    decoded_outputs = model.generate(
        input_map['input_ids'],
        do_sample=True,
        top_k=10,
        top_p=0.95,
        num_return_sequences=1
    )

    return tokenizer.decode(decoded_outputs[0])

if __name__ == "__main__":
    parser = ArgumentParser(parents=[build_parser()])
    args = parser.parse_args()

    sent = 'فردا یک قرار کاری با علی دارم'
    model, tokenizer = load_model(args)
    print(do_inference(model, tokenizer, sent))




