from transformers import AutoTokenizer
from hf_dataset import load_hf_ds_from_csv
from tqdm import tqdm
from hazm import Normalizer
from argparse import ArgumentParser
from text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)


class FarsInstructDataset:
    def __init__(self, tokenizer, max_len: int, split: str):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

        self.raw_data = load_hf_ds_from_csv()

    def tokenize_fn(self, args):
        df = self.raw_data[self.split].to_pandas()
        df = df.dropna()
        df.reset_index()
        df = df[args.start : args.end]
        print('Preprocessing input')
        new_df = preprocess_wo_hazm(df)

        print('Tokenizing dataset...')
        input_ids = []; attn_mask = []
        for item, row in tqdm(new_df.iterrows(), total=len(new_df)):
            encoded_data = self.tokenizer(
                '<s>' + row['prompt'] + '</s>',
                max_length=self.max_len,
                truncation=True,
                padding='max_length')
            
            input_ids.append(encoded_data['input_ids'])
            attn_mask.append(encoded_data['attention_mask'])

        new_df['input_ids'] = input_ids
        new_df['attn_mask'] = attn_mask

        
        return new_df
        
        
def preprocess(data_df):
    def hazm_normalization(text):   
        return text
    
    tqdm.pandas()
    
    data_df['prompt'] = data_df['inputs'] + '<|startoftext|>' + data_df['outputs'].progress_apply(lambda x: hazm_normalization(x))
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    return data_df

def preprocess_wo_hazm(data_df):
    def normalization(text):   
        # normalizer = Normalizer(persian_numbers=False)
        # return normalizer.normalize(text)
        text = text.replace("[n]", "\n")

        text = "".join(map_to_persian(char) for char in text)
        # Split the text into sentences
        snt = ''
        for sentence in split_into_sentences(text):
            # Remove the remaining punctuation
            sentence = patterns["ELIMINATE"].sub(" ", sentence)
            # Making sure there's a space after each comma
            sentence = patterns["COMMA"].sub("، ", sentence)
            # Multiple spaces into one
            sentence = patterns["TOO_MANY_SPACES"].sub(" ", sentence)
            # Strip the leading and the trailing white spaces
            sentence = sentence.strip()
            # Remove the spaces before punctuations
            sentence = patterns["NO_SPACE_BEFORE"].sub("", sentence)

            snt += sentence

        return snt
    
    data_df['prompt'] = data_df['inputs'].apply(lambda x: normalization(x)) + '<|startoftext|>' + data_df['outputs'].apply(lambda x: normalization(x))
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    return data_df


def concurrent_preprocess(data_df):
    import concurrent.futures
    
    def hazm_normalization(text):
        normalizer = Normalizer(persian_numbers=False)
        return normalizer.normalize(text)
        
    def apply_normalization(text):
        return hazm_normalization(text)

    num_threads = 8  # Choose the number of threads you want to use
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    total_items = len(data_df)
    progress_bar = tqdm(total=total_items)

    def norm_fn(x):
        progress_bar.update(1)
        return apply_normalization(x)

    data_df['normalized_outputs'] = list(executor.map(norm_fn, data_df['outputs']))
    data_df['normalized_inputs'] = list(executor.map(norm_fn, data_df['inputs']))

    executor.shutdown()

    data_df['prompt'] = data_df['normalized_inputs'] + '<|startoftext|>' + data_df['normalized_outputs']

    progress_bar.close()

    return data_df



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()

    model_path = 'HooshvareLab/gpt2-fa'
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              bos_token='<s>', 
                                              eos_token='</s>', 
                                              pad_token='<pad>')
   
    ds = FarsInstructDataset(tokenizer, max_len=1024, split='train')
    df = ds.tokenize_fn(args)
    df.to_csv(f'data/normalized_data_chuncks/tokenized_dataset{args.start}_{args.end}.csv')

