from transformers import GPT2LMHeadModel
import torch
from torch import nn


class DecoderModel(nn.Module):
    def __init__(self, model_name_or_path: str, config, **kwargs):
        """
        DecoderModel is the T0 approach to evaluate the model on labeling tasks.
        More details on: https://github.com/bigscience-workshop/t-zero/blob/master/t0/model.py
        """
        super(DecoderModel, self).__init__()
        self._model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, **kwargs)

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
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        predictions = seq_log_prob.argmax(dim=-1)
        return predictions