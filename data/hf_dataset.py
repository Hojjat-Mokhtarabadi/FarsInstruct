import os
import csv
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import pkg_resources

_DATA_PATH = Path('data')
_MAIN_PATH = Path(pkg_resources.resource_filename(__name__, 'dataset.csv'))
# _MAIN_PATH = Path(os.path.join('data', 'dataset.csv'))
_all_files = list(_DATA_PATH.rglob('*.*'))

def read_all_and_convert_t0_csv():
    _all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'type'])
    for file in _all_files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            # print(len(df))
            _all_dfs = pd.concat([_all_dfs, df])
            # print(len(_all_dfs))

    _all_dfs = _all_dfs.drop(columns=['Unnamed: 0'])
    _all_dfs.to_csv('data/dataset.csv', index=False, header=True)

def explore_data():
    df = pd.read_csv('dataset.csv')
    print(df.sample(10), 
          df['inputs'][0], 
          len(df))
    print(df[df['type'] == 'fs'])


def load_hf_ds_from_csv():
    path = str(_MAIN_PATH.resolve())
    dataset = load_dataset('csv', data_files=path)
    return dataset

    
if __name__ == "__main__":
    read_all_and_convert_t0_csv()
    # dataset = load_hf_ds_from_csv()
    # print(dataset.column_names)
    # test()
