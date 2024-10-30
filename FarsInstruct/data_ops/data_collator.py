from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union, Any, Dict, List, Tuple
import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
import numpy as np
from transformers import TextDataset, DataCollatorForLanguageModeling



@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        try: 
            num_choices = len(features[0]["input_ids"])
        
            flattened_features = [
                [
                    {
                        k: v[i]
                        for k, v in feature.items()
                        if k != "targets"
                    }
                    for i in range(num_choices)
                ]
                for feature in features
            ]
            flattened_features = list(chain(*flattened_features))
        except:
            print(features[0])
            print(num_choices)


        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id]*(max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0]*(max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch



RESPONSE_KEY = f"### Response:\n"

class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        # The torch_call method overrides the same method in the base class and 
        # takes a list of examples as input.  
        batch = super().torch_call(examples)

        labels = batch["labels"].clone()

        # The code then encodes a special token, RESPONSE_KEY_NL, 
        # representing the end of the prompt followed by a newline. 
        # It searches for this token in the sequence of tokens (labels) 
        # and finds its index.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)


        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                # If the response token is not found in the sequence, it raises a RuntimeError. 
                # Otherwise, it determines the end index of the response token.
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs \
                    {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # To train the model to predict only the response and ignore the prompt tokens, 
            # it sets the label values before the response token to -100. 
            # This ensures that those tokens are ignored by the PyTorch loss function during training.
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

