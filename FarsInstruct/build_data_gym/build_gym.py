from data_gym import DataGym
from argparse import ArgumentParser
from argparse import ArgumentParser
from FarsInstruct.build_data_gym.utils import *
from FarsInstruct.build_data_gym.meta_data_handler import generate_meta_data_file

_MIN_SAMPLES = {
    "persiannlp/parsinlu_translation_fa_en": 50000,
    "persiannlp/parsinlu_translation_en_fa": 50000,
    "pn_summary": 60000,
    "wiki_summary": 60000,
    "PNLPhub/C-ExaPPC": 60000,
    "SajjadAyoubi/persian_qa": 60000,
    "PNLPhub/Persian-News": 60000,
    "persiannlp/parsinlu_sentiment": 60000,
    "persian_ner": 60000,
    "SLPL/syntran-fa": 60000,
    "PNLPhub/parsinlu-multiple-choice": 60000,
    "PNLPhub/FarsTail": 60000,
    "PNLPhub/DigiMag": 60000,
    "PNLPhub/Pars-ABSA": 60000,
    "PNLPhub/digikala-sentiment-analysis": 60000,
    "persiannlp/parsinlu_query_paraphrasing": 60000,
    "persiannlp/parsinlu_entailment": 60000,
    "parsinlu_reading_comprehension":60000,
    "PNLPhub/snappfood-sentiment-analysis": 60000
}
        
def build_gym(ds_name: str = 'all', split: str = 'train', shots: int = 3):
    def retrieve_prompt_and_apply(item):
        """
        Retrieve the jinja templates and apply them on the given dataset.
        """
        for template_name in item['Prompt names']:
                print('Generating data based on {}_{}...'.format(item['Dataset name'], template_name))
                subset_name = item['Subset']
                try:
                    if shots > 1:
                        data_gym = DataGym(dataset_name, subset_name, template_name, split=split, min_samples=_MIN_SAMPLES[dataset_name])
                        data_gym.build_fs_gym(shots=shots)
                    elif shots == 1:
                        data_gym = DataGym(dataset_name, subset_name, template_name, split=split, min_samples=_MIN_SAMPLES[dataset_name])
                        data_gym.build_zs_gym()
                except Exception as e:
                    print(e)
                else:
                    continue

    prompted_datasets = load_prompted_datasets()
    ds_names = ds_name.split(',')
    for item in prompted_datasets: 
        dataset_name = item['Dataset name']
        
        if ds_name == 'all':
            retrieve_prompt_and_apply(item)
        else: 
            if dataset_name in ds_names:
                    retrieve_prompt_and_apply(item)


if __name__ == "__main__":
    parser = ArgumentParser("Data gym builder")
    parser.add_argument('--ds_name', default='all', type=str)
    parser.add_argument('--shots', default=3, type=int)
    parser.add_argument('--split', required=True, choices=['train', 'validation', 'test'])
    parser.add_argument('--generate_metadata', action='store_true')
    parser.add_argument('--build_gym', action='store_true')
    parser.add_argument('--gym_to_csv', action='store_true')
    args = parser.parse_args()

    print(f'#of shots {args.shots}')

    if args.build_gym:
        build_gym(ds_name=args.ds_name, split=args.split, shots=args.shots)
    if args.gym_to_csv:
        path =  f'data/{args.shots}shot_instruct_dataset_test_all.csv'
        read_all_and_convert_t0_csv(ds_name=args.ds_name, split=args.split, shots=args.shots, output_path=path)

    print(f"Generate metadata file: {args.generate_metadata}")
    if args.generate_metadata:
        generate_meta_data_file()


