from data_gym import DataGym
from argparse import ArgumentParser
from argparse import ArgumentParser
from FarsInstruct.build_data_gym.utils import *
from FarsInstruct.build_data_gym.meta_data_handler import generate_meta_data_file
        
def build_gym(ds_name: str = 'all', split: str = 'train'):
    def retrieve_prompt_and_apply(item):
        """
        Retrieve the jinja templates and apply them on the given dataset.
        - for the sake of convenience we differentiated zero-shot or few-shot tasks by their name
        """
        for template_name in item['Prompt names']:
                print('Generating data based on {}_{}...'.format(item['Dataset name'], template_name))
                try:
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


        
if __name__ == "__main__":
    parser = ArgumentParser("Data gym builder")
    parser.add_argument('--ds_name', default='all')
    parser.add_argument('--split', required=True, choices=['train', 'validation', 'test'])
    parser.add_argument('--generate_metadata', action='store_true')
    args = parser.parse_args()

    build_gym(ds_name=args.ds_name, split=args.split)
    read_all_and_convert_t0_csv(split=args.split)

    print(f"Generate metadata file: {args.generate_metadata}")
    if args.generate_metadata:
        generate_meta_data_file()


