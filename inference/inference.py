from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_PATH = "HooshvareLab/gpt2_fa"

def do_inference(txt):
    model = GT2LMHeadModel.from_pretrained(Model_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, 
                                              bos_token='<s>', 
                                              eos_token='</s>', 
                                              pad_token='<pad>')
    

    ids = tokenizer(txt)
    model.generate(ids)



if __name__ == "__main__":
    do_inference()
