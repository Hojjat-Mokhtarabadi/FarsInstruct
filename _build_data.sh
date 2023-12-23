cd FarsInstruct

python build_data_gym/build_gym.py --split test \
                                   --ds_name PNLPhub/FarsTail \
                                   --generate_metadata \
                                   --prompt_format hooshvare

cd ..
