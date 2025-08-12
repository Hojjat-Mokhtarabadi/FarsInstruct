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
                "Subset": subset_name,
                "Number of prompts": len(dataset_templates),
                "Prompt names": [t.name for t in dataset_templates.templates.values()],
            }
        )

    return results

   
def read_all_and_convert_t0_csv(ds_name, split, shots, output_path):
    """
    Read all generated datasets and concatinate them to insturct_dataset
    """
    ds_path = output_path
    ds_names = ds_name.split(',')
    all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'ds', 'template'])
    all_files = list(DATA_PATH.rglob('*.*'))
    for file in tqdm(all_files, total=len(all_files)):
        if file.parent.parent.parent.name != 'data':
            file_parent_name = f"{file.parent.parent.parent.name}/{file.parent.parent.name}"
        else: 
            file_parent_name = file.parent.parent.name

        if 'sample_data' in file.parent.name:
            continue
       # if ds_name != 'all' and file_parent_name not in ds_names:
       #     continue
        
        if file.name.endswith('.csv') and split == file.parent.name:
            print(file.parent.parent.name)
            df = pd.read_csv(file)
            all_dfs = pd.concat([all_dfs, df])

    try:
        all_dfs = all_dfs.drop(columns=['Unnamed: 0'])
    except:
        print('Unnamed: 0 does not exist!')

    all_dfs = all_dfs.dropna()
    all_dfs.to_csv(ds_path, index=False, header=True, mode='w+')
    print(f"Dataset save as: {ds_path}")
    print('Done!')
