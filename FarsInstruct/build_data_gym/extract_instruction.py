from promptsource.templates import DatasetTemplates
from datasets import load_dataset
from FarsInstruct.build_data_gym.utils import *
        
def do_extraction(ds_name: str = 'all', split: str = 'train'):
    def retrieve_prompt_and_apply(item):
        """
        Retrieve the jinja templates and apply them on the given dataset.
        """
        lst = []
        dss = []
        for template_name in item['Prompt names']:
                print('Generating data based on {}_{}...'.format(item['Dataset name'], template_name))
                subset_name = item['Subset']
                try:
                    inst = extract_instruction(dataset_name, subset_name, template_name, split=split)
                    lst.append(inst)
                    dss.append(item['Dataset name'])
                except Exception as e:
                    try :
                        inst = extract_instruction(dataset_name, subset_name, template_name, split='test')
                        lst.append(inst)
                        dss.append(item['Dataset name'])
                    except:
                         print(e)
                else:
                    continue

        return lst, dss

    prompted_datasets = load_prompted_datasets()
    ds_names = ds_name.split(',')
    all_inst = []
    all_ds = []
    for item in prompted_datasets: 
        dataset_name = item['Dataset name']
        
        if ds_name == 'all':
            r = retrieve_prompt_and_apply(item)
        else: 
            if dataset_name in ds_names:
                    r = retrieve_prompt_and_apply(item)

        all_inst += r['inst']
        all_ds += r['dss']
        

    print(all_inst)

    dd = pd.DataFrame({'instructions': all_inst, 'datasets': all_ds})

    dd.to_excel('data/instructions.xlsx')

    return dd

    

        
def extract_instruction(dataset_name:str, subset_name: str, template_name: str, split: str):        
    dataset_name = dataset_name
    subset_name = subset_name
    data = load_dataset(dataset_name, split=split)
    split = split
    template_name = template_name
    template = DatasetTemplates(dataset_name, subset_name)[template_name]

    def remove_instruction(x):
        splt_text = x.split('\n\n')
        return '\n'.join(splt_text[1:]), splt_text[0]

    result = template.apply(data[2])  
    input_wo_instruct, instruction = remove_instruction(result[0])

    return instruction
            

   
if __name__ == "__main__":
    do_extraction()


