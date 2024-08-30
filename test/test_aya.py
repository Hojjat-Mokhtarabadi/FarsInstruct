# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

repo = '/media/abbas/Backup/aya'

tokenizer = AutoTokenizer.from_pretrained(repo)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")


# Turkish to English translation
tur_inputs = tokenizer.encode("Translate to English: Aya cok dilli bir dil modelidir.", return_tensors="pt")
tur_outputs = aya_model.generate(tur_inputs, max_new_tokens=128)
print(tokenizer.decode(tur_outputs[0]))
# Aya is a multi-lingual language model

# Q: Why are there so many languages in India?
hin_inputs = tokenizer.encode("भारत में इतनी सारी भाषाएँ क्यों हैं?", return_tensors="pt")
hin_outputs = aya_model.generate(hin_inputs, max_new_tokens=128)
print(tokenizer.decode(hin_outputs[0]))