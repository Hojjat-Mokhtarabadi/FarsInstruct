import pandas as pd
from promptsource.templates import TemplateCollection
from tqdm import tqdm
from pathlib import Path

DATA_PATH = Path('data')
# _MAIN_PATH = Path(os.path.join('data', 'dataset.csv'))

def load_prompted_datasets():
    """
    Load all persian prompted datasets
    """
    template_collection = TemplateCollection(language='fa')
    results = []
    for (dataset_name, subset_name) in template_collection.keys:
        dataset_templates = template_collection.get_dataset(dataset_name, subset_name, 'fa')
        results.append(
            {
                "Dataset name": dataset_name,
                "Number of prompts": len(dataset_templates),
                "Prompt names": [t.name for t in dataset_templates.templates.values()],
            }
        )

    return results

   
def read_all_and_convert_t0_csv(split):
    """
    Read all generated datasets and concatinate them to insturct_dataset
    """
    all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'type', 'ds', 'template'])
    all_files = list(DATA_PATH.rglob('*.*'))
    for file in tqdm(all_files, total=len(all_files)):
        if 'sample_data' in file.parent.name:
            continue
        
        if file.name.endswith('.csv') and split == file.parent.name:
            df = pd.read_csv(file)
            all_dfs = pd.concat([all_dfs, df])

    all_dfs = all_dfs.drop(columns=['Unnamed: 0'])
    all_dfs = all_dfs.dropna()
    all_dfs.to_csv(f'data/instruct_dataset_{split}.csv', index=False, header=True, mode='w+')
    print('Done!')