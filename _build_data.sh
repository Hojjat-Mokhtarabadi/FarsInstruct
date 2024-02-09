#/bin/bash

cd FarsInstruct

#start building
for i in 'test' 'train' 'validation'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name 'all' \
                                   --prompt_format llama \
                                   --generate_metadata

done
cd ..
