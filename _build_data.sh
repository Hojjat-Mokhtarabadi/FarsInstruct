#/bin/bash
cd FarsInstruct

#start building
for i in 'train' 'test' 'validation'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name "all" \
				   --build_gym \
                                   --generate_metadata \
                                   --shots 1


done
cd ..
