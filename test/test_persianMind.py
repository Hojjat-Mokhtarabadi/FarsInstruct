from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# model_path = "FarsInstruct/results/hf_ckpt_macro_persianmind_exa_300"
# model_path = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/hf_ckpt_macro_mistral_exa_all"
# model_path = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/hf_ckpt_macro_mistral_exa_pn_sum_digi_absa"
# model_path = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/hf_ckpt_macro_mistral_exa_pn_sum"
# model_path = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/hf_ckpt_macro_mistral_exa_pn_sum_wiki.with_base.pn_sum_exa"
# model_path = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki"
# model_path = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran.BASE.pnsum_wiki"
# model_path = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran_exa.BASE.pnsum_wiki_syntran"
# model_path = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran_exa_absa.BASE.pnsum_wiki_syntran_exa"
model_path = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran_exa_absa_qa.BASE.pnsum_wiki_syntran_exa_absa.checkpoint-1300"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # "/media/abbas/Backup/Mistral-7B-Instruct-v0.2",
    #"/media/abbas/Backup/aya/",
    # "/media/abbas/Backup/PersianMind-v1.0",
    # torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    torch_dtype=torch.float16, 
    device_map="auto"
)
# model.pad
tokenizer = AutoTokenizer.from_pretrained(
    # "/media/abbas/Backup/PersianMind-v1.0",
    # "/media/abbas/Backup/Mistral-7B-Instruct-v0.2",
    model_path
    #"/media/abbas/Backup/aya/",
)

TEMPLATE = "{context}{prompt}"
CONTEXT = "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of " \
    "NLP experts at the University of Tehran to help you with various tasks such as answering questions, " \
    "providing recommendations, and helping with decision making. You can ask it anything you want and " \
    "it will do its best to give you accurate and relevant information."

raw_data = load_dataset('csv', data_files={'train': 'FarsInstruct/data/1shot_farsintruct_p3_train.csv'}, split='train')
ds_name = ["pn_summary"]
ds_list = []
for ds in ds_name:
    min_chunk = 20  
    raw_data_filterd = raw_data.filter(lambda ex: ex["ds"] == ds) 
    raw_data_filterd = raw_data_filterd.shuffle().select(range(0, min(min_chunk, len(raw_data_filterd))))
    ds_list.append(raw_data_filterd)
    

ds = concatenate_datasets(ds_list)


# all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'ds', 'template'])
# inputs = []
# outputs = []
# preds = []
# for i in df['ds'].unique():
#     min_chunk = 10
#     dff = df[df['ds'] == i]
#     for j in dff['template'].unique():
#         df_item = dff[dff['template'] == j]
#         spl = df_item.sample(min_chunk).reset_index()

#         inputs.append(spl['inputs'][:])
#         outputs.append(spl['outputs'][:])


# PROMPT = PROMPT.replace("[n]", '\n')
# model_input = TEMPLATE.format(context='', prompt=PROMPT)
# input_tokens = tokenizer(ds, return_tensors="pt")
input_tokens = tokenizer.batch_encode_plus(ds, return_tensors="pt")
input_tokens = input_tokens.to(device)
generate_ids = model.generate(**input_tokens, max_new_tokens=512)
model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(model_output)
# print(model_output[len(model_input):])