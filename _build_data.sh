#/bin/bash
cd FarsInstruct

#start building
for i in 'test' 'train' 'validation'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name "PNLPhub/parsinlu-multiple-choice,PNLPhub/snappfood-sentiment-analysis,PNLPhub/digikala-sentiment-analysis" \
                                   --generate_metadata \
                                   --shots 3


done
cd ..
