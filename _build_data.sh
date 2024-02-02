#/bin/bash

cd FarsInstruct

#start building
for i in 'test' 'train' 'validation'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name PNLPhub/FarsTail \
                                   --generate_metadata \
                                   --prompt_format llama 

done
cd ..
