from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

repo_name = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/mistral-persianmind.train-on-col-2-3.eval-on-col-1-shot-1-v2/checkpoint-7000"

#repo_name = "/home/hojjat/workstation/FarsInstruct/FarsInstruct/results/persian_mind.snapp-digi-pn_sum-syntran-qa-pharaphrase-reading_comp-parsi_sent/checkpoint-6200"
config = PeftConfig.from_pretrained(repo_name)

model = AutoModelForCausalLM.from_pretrained(
    # config.base_model_name_or_path,
    "/media/abbas/Backup/Mistral-7B-Instruct-v0.2",
    #"/media/abbas/Backup/aya/",
    #"/media/abbas/Backup/PersianMind-v1.0",
    # torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    torch_dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    #"/media/abbas/Backup/Mistral-7B-Instruct-v0.2",
    "/media/abbas/Backup/PersianMind-v1.0",
    #"/media/abbas/Backup/aya/",
)

model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, repo_name)

TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
CONTEXT = "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of " \
    "NLP experts at the University of Tehran to help you with various tasks such as answering questions, " \
    "providing recommendations, and helping with decision making. You can ask it anything you want and " \
    "it will do its best to give you accurate and relevant information."
PROMPT = """
با توجه به متن داده شده آیا میتوان عبارت را نتیجه گرفت؟
- بله
- خیر
- شاید


متن: یا آنها را می شناسم یا نمی دانم که چطور ممکن است کسی گول این را بخورد که قرار است پول اضافی دریافت کند
عبارت: من بارها و بارها در تله کلاهبرداری آنها که میگویند «پول نقد اضافی خواهید گرفت» افتاده ام.
جواب:خیر


متن: این تجهیزات با گذر زمان فرسوده خواهند شد و به احتمال زیاد پیش از آغاز هر گونه اقدام جدی برای اسکان انسان در مریخ، مدارگردهای دیگری که مجهز به دستگاههای رلۀ M باشند در مدار این سیاره قرار داده خواهند شد. 
عبارت: کارگذاشتن دستگهای مخابراتی در مدار مریخ از مدتها پیش انجام شده است.
جواب:شاید


متن: و تابستان بود که تهویه هوا روشن بود و در بسته بود و من نمی توانستم در بزنم چون مجبور بودم جک را با دست دیگر نگه دارم. بالاخره با آرنجم زنگ در را زدم و مادر به سراغ در آمد
عبارت: زمستان بود و تهویه مطبوع روشن بود ، من نمی توانستم زنگ در را زنگ بزنم زیرا یخ زده بود.
:جواب
"""

model_input = TEMPLATE.format(context='', prompt=PROMPT)
input_tokens = tokenizer(model_input, return_tensors="pt")
input_tokens = input_tokens.to(device)
generate_ids = model.generate(**input_tokens, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(model_output)
print(model_output[len(model_input):])