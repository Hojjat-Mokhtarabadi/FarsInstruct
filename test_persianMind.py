from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained(
    "/media/abbas/Backup/PersianMind-v1.0",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map={"": device},
)
tokenizer = LlamaTokenizer.from_pretrained(
    "/media/abbas/Backup/PersianMind-v1.0",
)

TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
CONTEXT = "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of " \
    "NLP experts at the University of Tehran to help you with various tasks such as answering questions, " \
    "providing recommendations, and helping with decision making. You can ask it anything you want and " \
    "it will do its best to give you accurate and relevant information."
PROMPT = "در مورد هوش مصنوعی توضیح بده."

model_input = TEMPLATE.format(context=CONTEXT, prompt=PROMPT)
input_tokens = tokenizer(model_input, return_tensors="pt")
input_tokens = input_tokens.to(device)
generate_ids = model.generate(**input_tokens, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(model_output[len(model_input):])