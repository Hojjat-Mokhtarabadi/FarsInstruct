from enum import Enum

class Phase(Enum):
    INSTRUCTION_TUNING = 0
    LABELING_EVALUATION = 1

# from dataclasses import dataclass
# from FarsInstruct.modeling import DecoderModel
# from transformers import GPT2LMHeadModel


# @dataclass
# class Phase:
#     INSTRUCTION_TUNING: dict = {
#         'base_model': GPT2LMHeadModel,
#         'encode_fn': "pretraining_encode_fn"
#         } 
#     LABELING_EVALUATION: dict = {
#         'base_model': DecoderModel,
#         'encode_fn': "encode_fn_based_on_t0"
#         } 
  
    