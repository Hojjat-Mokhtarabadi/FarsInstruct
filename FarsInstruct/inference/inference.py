from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM

BASE_MODELS = [
    '../checkpoints/hooshvare.zs-fs.digi-snapp.-1.bs8'
]

def do_inference(txt):
    for model in BASE_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model, 
                                            bos_token='<s>', 
                                            eos_token='</s>', 
                                            pad_token='<pad>')
        model = AutoModelForCausalLM.from_pretrained(model)
        model.resize_token_embeddings(len(tokenizer))

        ids = tokenizer(txt, return_tensors='pt')
        sent = model.generate(**ids, 
                              num_return_sequences=1,
                              max_new_tokens=10)
        
        print(tokenizer.decode(*sent))



if __name__ == "__main__":
    #prompt = 'احساس جمله خیلی جنس بدی بود چیست؟'
    prompt = 'سوال نظر شما در مورد سلامت و ارزش غذایی این شیر چیست؟ در چه دسته بندی قرار میگیرد؟'
    prompt = '''متن داده شده خوشحال هست یا ناراحت؟

متن: استفاده از مواد اولیه مرغوب و تازه کیفیت پیتزا رو فوق العاده کرده بود حجم هم عالی بود. واقعا ممنونم
نقد:خوشحال

متن: غذا کیفیت قابل قبولی داشت و توسط اسنپ اکسپرس هم به صورت گرم تحویل شد
نقد:خوشحال

متن: حجم غذا خیلی خوب بود. خوشمزه هم بود. ممنون
نقد:<|startoftext|>'''

    prompt = '''این سوال چه مواردی را مورد پرسش قرار داده است؟ 

سوال: نظر شما در مورد گریم، طراحی صحنه و جلوه های ویژه ی بصری فیلم  گرگ بازی چیست؟صحنه‌ها و محیط

سوال: نظر شما به صورت کلی در مورد فیلم  گرگ بازی چیست؟نکات کلی

سوال: نظر شما در مورد صداگذاری و جلوه های صوتی فیلم  قندون جهیزیه چیست؟<|startoftext|>'''
    

    do_inference(prompt)
