cd FarsInstruct

python build_data_gym/build_gym.py --split validation \
                                   --ds_name persiannlp/parsinlu_query_paraphrasing \
                                   --llama_compatible

cd ..
