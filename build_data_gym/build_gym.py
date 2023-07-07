from data_gym import DataGym
from argparse import ArgumentParser
from promptsource.templates import TemplateCollection

        
def load_prompted_datasets():
    template_collection = TemplateCollection(language='fa')
    print(template_collection.keys)
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


def build_gym():
    prompted_datasets = load_prompted_datasets()

    print(prompted_datasets)
    for item in prompted_datasets: 
        dataset_name = item['Dataset name']
        for template_name in item['Prompt names']:
            print('Generating data based on {}...'.format(template_name))

            if template_name[-2:] == "fs":
                data_gym = DataGym(dataset_name, template_name, split='train', type='fs', shots=3)
                data_gym()
            elif template_name[-2:] == "zs":
                data_gym = DataGym(dataset_name, template_name, split='train', type='zs', shots=1)
                data_gym()
            else:
                continue


if __name__ == "__main__":
    build_gym()