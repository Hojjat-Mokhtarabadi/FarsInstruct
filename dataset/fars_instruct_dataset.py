from torch.utils.data import Dataset
from data.hf_dataset import load_hf_ds_from_csv
from tqdm import tqdm
import pandas as pd
from .text_cleaning import (map_to_persian, 
                           split_into_sentences, 
                           patterns)


class FarsInstructDataset(Dataset):
    def __init__(self, tokenizer, max_len: int, split: str):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

        self.raw_data = load_hf_ds_from_csv()
        self.input_ids, self.attn_mask = self._tokenize_fn()


    def _tokenize_fn(self):
        df = self.raw_data[self.split].to_pandas()
        df = df[:5000]

        print('Preprocessing input')
        new_df = preprocess(df)

        print('Tokenizing dataset...')
        input_ids = []; attn_mask = []
        for item, row in tqdm(new_df.iterrows(), total=len(new_df)):
            encoded_data = self.tokenizer(
                '<s>' + row['prompt'] + '</s>',
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt')
            
            input_ids.append(encoded_data['input_ids'])
            attn_mask.append(encoded_data['attention_mask'])

        new_df['input_ids'] = input_ids
        new_df['attn_mask'] = attn_mask

        # new_df.to_csv('data/tokenized_dataset.csv')

        return input_ids, attn_mask
    
    def __len__(self):
        return len(self.raw_data) 

    def __getitem__(self, index: int):
        return self.input_ids[index], self.attn_mask[index]
        
        
def preprocess(data_df):
    def normalization(text):   
        # normalizer = Normalizer(persian_numbers=False)
        # return normalizer.normalize(text)
        text = text.replace("[n]", "\n")

        text = "".join(map_to_persian(char) for char in text)
        # Split the text into sentences
        snt = ''
        for sentence in split_into_sentences(text):
            # Remove the remaining punctuation
            # sentence = patterns["ELIMINATE"].sub(" ", sentence)
            # Making sure there's a space after each comma
            sentence = patterns["COMMA"].sub("، ", sentence)
            # Multiple spaces into one
            # sentence = patterns["TOO_MANY_SPACES"].sub(" ", sentence)
            # Strip the leading and the trailing white spaces
            sentence = sentence.strip()
            # Remove the spaces before punctuations
            sentence = patterns["NO_SPACE_BEFORE"].sub("", sentence)

            snt += sentence

        return snt

    tqdm.pandas()
    
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    data_df['prompt'] = data_df['inputs'].progress_apply(normalization) + '<|startoftext|>' + data_df['outputs'].apply(normalization)

    data_df = data_df.reset_index(drop=True)

    return data_df

if __name__ == "__main__":
    df = pd.DataFrame({
        'txt': ["""با توجه به سوال یک جواب کوتاه و یک جواب بلند بنویس

سوال: باشگاه هاکی ساوتهمپتون چه نام دارد؟
جواب کوتاه:"""]
    })
    new_df = preprocess(df)
    print(new_df['txt'].values)


