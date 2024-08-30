import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F402

# BASE_MODEL = "/media/abbas/Backup/PersianMind-v1.0"
#BASE_MODEL = "./base_checkpoints/llama3-8b-instruct"
#BASE_MODEL = "./lora_checkpoints/hf_ckpt_macro_llama3_parsinlu_enfa_faen_pnsum_wikisum_exappc_sajjadqa_persiannews"
#BASE_MODEL = "./lora_checkpoints/hf_ckpt_micro_ava_sajjadqa--3_pnsum--2_slpl--genqwitha-qora_wikisum--3_digi--whichctg_persiannews--chscatg_parssentiment--revasp-revcat_exap--rel-ordr_absa--plr_pner--fndprsn_readcomp--fndans-qc"

#BASE_MODEL = "./lora_checkpoints/hf_ckpt_micro_train_ava_sajjadqa--3_pnsum--2_slpl--3_wikisum--3_digi--whichctg_persiannews--2_parssentiment--revasp-revcat_exap--rel-ordr_absa--3_pner--fndprsn_reacomp--fndans-qc_trnsenfa--prven"
#  hf_ckpt_micro_ava_sajjadqa--qacatg-titlgen-genqWRa_pnsum--sumartcl-whatctgblngsto_slpl--genqwitha-qora_wikisum--artclsum-gnralans_digi--whichctg_persiannews--chscatg_parssentiment--revasp-revcat_exap--rel-ordr_absa--plr_pner--fndprsn_readcomp--fndans-qc"
BASE_MODEL = "./base_checkpoints/ava-llama3-v2"

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
    "./FarsInstruct/results/ava_raw_fine_tune_v2_/checkpoint-9950",
    #"./FarsInstruct/results/micro_train_ava_sajjadqa--3_pnsum--2_slpl--3_wikisum--3_digi--whichctg_persiannews--2_parssentiment--revasp-revcat_exap--rel-ordr_absa--3_pner--fndprsn_reacomp--fndans-qc_trnsenfa--prven_faen--trnsen/checkpoint-1110",
   # "./FarsInstruct/results/ava_raw_fine_tune_with_really_small_samples/checkpoint-870",
   # "./FarsInstruct/results/micro_train_ava_sajjadqa--3_pnsum--2_slpl--genqwitha-qora_wikisum--artclsum-gnralans_digi--whichctg_persiannews--chscatg_parssentiment--revasp-revcat_exap--rel-ordr_absa--plr_pner--fndprsn_reacomp--fndans-qc/checkpoint-780",
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

#dirr = "./lora_checkpoints/hf_ckpt_micro_train_ava_sajjadqa--3_pnsum--2_slpl--3_wikisum--3_digi--whichctg_persiannews--2_parssentiment--revasp-revcat_exap--rel-ordr_absa--3_pner--fndprsn_reacomp--fndans-qc_trnsenfa--prven_faen--trnsen"
#dirr = "./lora_checkpoints/hf_ckpt_ava_raw_fine_tune_with_really_small_samples"
dirr = "./lora_checkpoints/hf_ckpt_ava_raw_lora_finetuned"
print("Saving merged model...")
base_model.save_pretrained(
    dirr, state_dict=deloreanized_sd, max_shard_size="1000MB"
)

tokenizer.save_pretrained(dirr)
