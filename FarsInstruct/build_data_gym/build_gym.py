from data_gym import DataGym
from argparse import ArgumentParser
from promptsource.templates import TemplateCollection
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import json
from promptsource.templates import DatasetTemplates


DATA_PATH = Path('data')
# _MAIN_PATH = Path(os.path.join('data', 'dataset.csv'))
all_files = list(DATA_PATH.rglob('*.*'))

        
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


def build_gym(ds_name: str = 'all', split: str = 'train'):
    def retrieve_prompt_and_apply(item):
        """
        Retrieve the jinja templates and apply them on the given dataset.
        - for the sake of convenience we differentiated zero-shot or few-shot tasks by their name
        """
        for template_name in item['Prompt names']:
                print('Generating data based on {}_{}...'.format(item['Dataset name'], template_name))
                try:
                    save_meta_data(dataset_name, template_name)
                    if template_name[-2:] == "fs":
                        data_gym = DataGym(dataset_name, template_name, split=split, type='fs', shots=3)
                        data_gym()
                    elif template_name[-2:] == "zs":
                        data_gym = DataGym(dataset_name, template_name, split=split, type='zs', shots=1)
                        data_gym()
                except Exception as e:
                    print(e)
                else:
                    continue

    prompted_datasets = load_prompted_datasets()
    for item in prompted_datasets: 
        dataset_name = item['Dataset name']

        if ds_name == dataset_name:
            retrieve_prompt_and_apply(item)
        if ds_name == 'all':
            retrieve_prompt_and_apply(item)
        else: continue


def save_meta_data(dataset_name, template_name):
    template = DatasetTemplates(dataset_name)[template_name]
    meta_data = {
                'dataset': f'{dataset_name}', 
                'template': f'{template_name}',
                'lang': template.metadata.languages,
                'ans_choice':template.get_fixed_answer_choices_list(),
                'metrics': template.metadata.metrics
                }
   
    with open("data/metadata.json", 'a+', encoding='utf-8') as f:
        json.dump(meta_data, f)


def read_all_and_convert_t0_csv(split):
    """
    Read all generated datasets and concatinate them to insturct_dataset
    """
    all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'type', 'ds'])
    for file in tqdm(all_files, total=len(all_files)):
        if 'sample_data' in file.parent.name:
            continue
        
        if file.name.endswith('.csv') and split == file.parent.name:
            print(file.parent)
            print(file)
            df = pd.read_csv(file)
            all_dfs = pd.concat([all_dfs, df])

    all_dfs = all_dfs.drop(columns=['Unnamed: 0'])
    all_dfs = all_dfs.dropna()
    all_dfs.to_csv(f'data/instruct_dataset_{split}.csv', index=False, header=True, mode='w+')



        

if __name__ == "__main__":
    parser = ArgumentParser("Data gym builder")
    parser.add_argument('--ds_name', default='all')
    parser.add_argument('--split', required=True, choices=['train', 'validation', 'test'])
    args = parser.parse_args()

    build_gym(ds_name=args.ds_name, split=args.split)
    #read_all_and_convert_t0_csv(split=args.split)


