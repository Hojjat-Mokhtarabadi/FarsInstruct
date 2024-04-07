import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

BASE_MODEL = "/media/abbas/Backup/PersianMind-v1.0"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/persianmind.train-on-col-2-3.eval-on-col-1-shot-mix/checkpoint-10000",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)


lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, "FarsInstruct/results/fianl_hf_ckpt", state_dict=deloreanized_sd, max_shard_size="800MB"
)