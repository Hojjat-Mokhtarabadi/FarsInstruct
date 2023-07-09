import os
import csv
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import pkg_resources
from tqdm import tqdm

_DATA_PATH = Path('data')
_MAIN_PATH = Path(pkg_resources.resource_filename(__name__, 'instruct_dataset.csv'))
# _MAIN_PATH = Path(os.path.join('data', 'dataset.csv'))
_all_files = list(_DATA_PATH.rglob('*.*'))

def read_all_and_convert_t0_csv():
    """
    Read all generated datasets and concatinate them to insturct_dataset.csv
    """
    _all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'type'])
    for file in tqdm(_all_files, total=len(_all_files)):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            _all_dfs = pd.concat([_all_dfs, df])

    _all_dfs = _all_dfs.drop(columns=['Unnamed: 0'])
    _all_dfs.to_csv('data/instruct_dataset.csv', index=False, header=True)

def explore_data():
    df = pd.read_csv('instruct_dataset.csv')
    print(df.sample(10), 
          df['inputs'][0], 
          len(df))
    print(df[df['type'] == 'fs'])


def load_hf_ds_from_csv(split, streaming):
    path = str(_MAIN_PATH.resolve())
    dataset = load_dataset('csv', data_files=path, split=split, streaming=streaming)
    return dataset

    
if __name__ == "__main__":
    read_all_and_convert_t0_csv()
    # dataset = load_hf_ds_from_csv()
    # print(dataset.column_names)
    # test()
