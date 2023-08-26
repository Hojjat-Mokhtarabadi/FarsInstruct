import streamlit as st
import json
from datasets import concatenate_datasets, load_dataset, Features, Value

def prepare_dataset(dataset):
    ds = load_dataset('csv', data_files='data/instruct_dataset.csv', split='train')
    fs = ds.filter(lambda x: x['type'] == 'fs' and x['ds'] == dataset).shuffle(seed=10).select(range(1, 50))
    zs = ds.filter(lambda x: x['type'] == 'zs' and x['ds'] == dataset).shuffle(seed=10).select(range(1 ,50))

    ds = concatenate_datasets([zs, fs])
    ds.to_json('data/sample_data/{}_train.json'.format(dataset))
    
def main():
    st.title("FarsInstruct data exploration")
    # left_json_file = st.sidebar.file_uploader("Upload JSON for Left Side", type=["json"])
    # right_json_file = st.sidebar.file_uploader("Upload JSON for Right Side", type=["json"])

    datasets = ('SajjadAyoubi/persian_qa', 'pn_summary', 'SLPL/syntran-fa',
       'wiki_summary', 'persiannlp/parsinlu_entailment',
       'persiannlp/parsinlu_sentiment',
       'persiannlp/parsinlu_query_paraphrasing', 'PNLPhub/Persian-News',
       'PNLPhub/FarsTail', 'PNLPhub/snappfood-sentiment-analysis',
       'PNLPhub/parsinlu-multiple-choice',
       'PNLPhub/digikala-sentiment-analysis', 'PNLPhub/DigiMag')

    with st.sidebar:
        option = st.selectbox(
            "Select the dataset",
            datasets
            )
        prepare_dataset(option)
        
        data_split = st.selectbox(
            "Select data split", 
            ("train", "val", "test")
        )

    with st.sidebar:
        dd = {}
        dd['fs'] = st.checkbox("few-shot", key="disabled")
        dd['zs'] = st.checkbox("zero-shot", key="disabled")

    with open('data/sample_data/{}_train.json'.format(option), 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for k, v in dd.items():
                if v and data['type'] == k:
                    st.text(data['inputs'])
                    st.json(data)

            


if __name__ == "__main__":
    main()