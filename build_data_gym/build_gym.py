from data_gym import DataGym
from argparse import ArgumentParser
from promptsource.templates import TemplateCollection

        
def load_prompted_datasets():
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


def build_gym(ds_name: str = 'all'):
    def retrieve_prompt_and_apply(item):
        """
        Retrieve the jinja templates and apply them on the given dataset.
        - for the sake of convenience we differentiated zero-shot or few-shot tasks by their name
        """
        for template_name in item['Prompt names']:
                print('Generating data based on {}_{}...'.format(item['Dataset name'], template_name))
                if template_name[-2:] == "fs":
                    data_gym = DataGym(dataset_name, template_name, split='train', type='fs', shots=3)
                    data_gym()
                elif template_name[-2:] == "zs":
                    data_gym = DataGym(dataset_name, template_name, split='train', type='zs', shots=1)
                    data_gym()
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


if __name__ == "__main__":
    build_gym(ds_name='pn_summary')