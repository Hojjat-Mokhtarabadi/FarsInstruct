import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F402

# BASE_MODEL = "/media/abbas/Backup/PersianMind-v1.0"
# BASE_MODEL = "/media/abbas/Backup/Mistral-7B-Instruct-v0.2"
# BASE_MODEL = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/hf_ckpt_macro_mistral_exa_pn_sum"
# BASE_MODEL = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki"
# BASE_MODEL = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran.BASE.pnsum_wiki"
BASE_MODEL = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran_exa_absa.BASE.pnsum_wiki_syntran_exa"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_pn_sum_wiki_syntran_exa_absa_qa.BASE.pnsum_wiki_syntran_exa_absa/checkpoint-1300",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_pn_sum_wiki_syntran_exa_absa.BASE.pnsum_wiki_syntran_exa/checkpoint-1800",
    # "FarsInstruct/results/macro_train_mistral_pn_sum_wiki_syntran_exa.BASE.pnsum_wiki_syntran/checkpoint-1400",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_pn_sum_wiki_syntran.BASE.pnsum_wiki/checkpoint-1400",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_pn_sum_wiki/checkpoint-1500",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_exa_pn_sum_wiki.with_base.mistral_exa_pnsum/checkpoint-900",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_exa_pn_sum_digi/checkpoint-1500",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_exa_pn_sum/checkpoint-900",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_mistral_exa_all/checkpoint-900",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/macro_train_persianmind_exa_different_point/checkpoint-300",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/persianmind-pn_sum-syntran-qa-exa-reading_comp/checkpoint-4500",
    # "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/mistral.train-on-col-2-3.eval-on-col-1-shot-mix/checkpoint-3000",
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

dirr = "/media/abbas/Backup/hf_ckpt_macro_mistral_pn_sum_wiki_syntran_exa_absa_qa.BASE.pnsum_wiki_syntran_exa_absa.checkpoint-1300"

base_model.save_pretrained(
    dirr, state_dict=deloreanized_sd, max_shard_size="1000MB"
)

tokenizer.save_pretrained(dirr)