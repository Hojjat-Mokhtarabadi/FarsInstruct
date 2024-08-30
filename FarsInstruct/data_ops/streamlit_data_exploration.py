import streamlit as st
import json
from datasets import concatenate_datasets, load_dataset 
from FarsInstruct.data_ops.paths import DATA_FILES

def prepare_dataset(dataset, split):
    ds = load_dataset('csv', data_files=DATA_FILES, split=split)

    fs = ds.filter(lambda x: x['type'] == 'fs' and x['ds'] == dataset).shuffle(seed=10).select(range(1, 50))
    zs = ds.filter(lambda x: x['type'] == 'zs' and x['ds'] == dataset).shuffle(seed=10).select(range(1 ,50))

    ds = concatenate_datasets([zs, fs])
    ds.to_json(f'data/sample_data/{dataset}_{split}.json')
    
def main():
    st.title("FarsInstruct data exploration")
    # left_json_file = st.sidebar.file_uploader("Upload JSON for Left Side", type=["json"])
    # right_json_file = st.sidebar.file_uploader("Upload JSON for Right Side", type=["json"])

    datasets = ('SajjadAyoubi/persian_qa', 'pn_summary', 'SLPL/syntran-fa',
       'wiki_summary', 'persiannlp/parsinlu_entailment',
       'persiannlp/parsinlu_sentiment',
       'persiannlp/parsinlu_query_paraphrasing', 'PNLPhub/Persian-News',
       'PNLPhub/FarsTail', 'PNLPhub/snappfood-sentiment-analysis',
       'PNLPhub/parsinlu-multiple-choice', 'parsinlu_reading_comprehension',
       'PNLPhub/digikala-sentiment-analysis', 'PNLPhub/DigiMag')
    
    with st.sidebar:
        selected_dataset = st.selectbox(
            "Select the dataset",
            datasets
        )   
        data_split = st.selectbox(
            "Select data split", 
            ("train", "validation", "test")
        )

        prepare_dataset(selected_dataset, data_split)

    with st.sidebar:
        dd = {}
        dd['fs'] = st.checkbox("few-shot", key="disabled")
        dd['zs'] = st.checkbox("zero-shot", key="disabled")

    with open(f'data/sample_data/{selected_dataset}_{data_split}.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for k, v in dd.items():
                if v and data['type'] == k:
                    st.text(data['inputs'])
                    st.json(data)

            


if __name__ == "__main__":
    main()