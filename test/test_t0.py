# Load an example from the datasets ag_news
from datasets import load_dataset
dataset = load_dataset("anli", split="train_r1")
example = dataset[1]

# Load prompts for this dataset
from promptsource.templates import DatasetTemplates
ag_news_prompts = DatasetTemplates('anli', lanuguage='en')

# Select a prompt by its name
prompt = ag_news_prompts["guaranteed/possible/impossible"]

# # Apply the prompt to the example
result = prompt.apply(example)
# print("INPUT: ", result[0])


from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

repo_name = "/media/abbas/Backup/Mistral-7B-Instruct-v0.2"
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# repo_name = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/mistral-persianmind.train-on-col-2-3.eval-on-col-1-shot-1-v2/checkpoint-7000"
# config = PeftConfig.from_pretrained(repo_name)

model = AutoModelForCausalLM.from_pretrained(
    # config.base_model_name_or_path,
    # "/media/abbas/Backup/Mistral-7B-Instruct-v0.2",
    # "/media/abbas/Backup/PersianMind-v1.0",
    # torch_dtype=torch.bfloat16,
    repo_name,
    quantization_config=quantization_config,
    torch_dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(repo_name)

# prompt = "My favourite condiment is"

# model_input = TEMPLATE.format(context='', prompt=PROMPT)
model_input = result[0]
input_tokens = tokenizer(model_input, return_tensors="pt")
input_tokens = input_tokens.to(device)
generate_ids = model.generate(**input_tokens, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(model_output)
print(model_output[:])
print(10 * '-')
print(result[1])

