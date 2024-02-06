from promptsource.templates import DatasetTemplates
from .utils import load_prompted_datasets
import json


def get_meta_data(dataset_name, subset_name, template_name):
    template = DatasetTemplates(dataset_name, subset_name)[template_name]
    meta_data = {
                'template': f'{template_name}',
                'lang': template.metadata.languages,
                'ans_choice':template.get_fixed_answer_choices_list(),
                'metrics': template.metadata.metrics,
                'choice_in_temp': template.metadata.choices_in_prompt
                }
    
    return meta_data


def generate_meta_data_file():
    meta_data_dict = {}
    prompted_datasets = load_prompted_datasets()
    for item in prompted_datasets: 
        dataset_name = item['Dataset name']
        subset_name = item['Subset']
        meta_data_list = []
        for template_name in item['Prompt names']:
            meta_data_list.append(get_meta_data(dataset_name, subset_name, template_name))

        meta_data_dict[f'{dataset_name}'] = meta_data_list


    with open("data/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(meta_data_dict, f)
        print('Metadata created!')


