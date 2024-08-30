#/bin/bash
cd FarsInstruct

#start building
for i in 'test'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name "all" \
				   --gym_to_csv \
                                   --generate_metadata \
                                   --shots 1


done
cd ..
