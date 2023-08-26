from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM

BASE_MODELS = [
    './checkpoints/hooshvare.zs.parsi_sentiment.15000.bs4',
    './checkpoints/bolbolzaban.zs.parsi_sentiment.15000'
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
        # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
    model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa")
    do_inference(prompt)
