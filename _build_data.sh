cd FarsInstruct

python build_data_gym/build_gym.py --split validation \
                                   --ds_name PNLPhub/FarsTail \
                                   --generate_metadata \
                                   --prompt_format llama

cd ..
