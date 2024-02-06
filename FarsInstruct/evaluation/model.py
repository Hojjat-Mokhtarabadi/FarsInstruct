import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftConfig, PeftModel

def load_causal_model(model_name_or_path, peft_model_id, current_model):
    if current_model != None:
        return current_model

    if peft_model_id:
        config = PeftConfig.from_pretrained(peft_model_id)
    else:
        config = AutoConfig.from_pretrained(model_name_or_path)

    _model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

    if peft_model_id:
        _model = PeftModel.from_pretrained(_model, peft_model_id)

    return _model


class DecoderModel(nn.Module):
    def __init__(self, model_name_or_path: str, peft_model_id = None, current_model = None):
        super(DecoderModel, self).__init__()
        
        if current_model != None:
            self._model = current_model
        else:
            if peft_model_id != None:
                config = PeftConfig.from_pretrained(peft_model_id)
            else:
                config = AutoConfig.from_pretrained(model_name_or_path)

            self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

            if peft_model_id != None:
                self._model = PeftModel.from_pretrained(self._model, peft_model_id)


    def forward(self, batch):
        device = batch["input_ids"].device
        _, prefix_length = batch["input_ids"].shape

        model_inputs = {
            "input_ids": torch.cat([batch["input_ids"], batch["labels"]], dim=-1),
            "attention_mask": torch.cat([batch["attention_mask"], batch["labels_attention_mask"]], dim=-1),
        }
        # Set position ids correctly to take care of padding tokens between inputs_ids and labels
        position_ids = torch.maximum(
            torch.cumsum(model_inputs["attention_mask"].to(torch.long), dim=-1) - 1,
            torch.zeros(1, dtype=torch.long, device=device)[None, None]
        )
        model_inputs["position_ids"] = position_ids

        logits = self._model(**model_inputs).logits[:, prefix_length-1:-1]
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),-1) 
        predictions = seq_log_prob.argmax(dim=-1)
        return predictions