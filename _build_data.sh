#/bin/bash
cd FarsInstruct

#start building
for i in 'validation' 'test'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name "all" \
                                   --generate_metadata \
                                   --shots 3


done
cd ..
