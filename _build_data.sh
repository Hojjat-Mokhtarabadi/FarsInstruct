#/bin/bash
cd FarsInstruct

#start building
for i in 'train'
do
echo "Building split: $i"

python build_data_gym/build_gym.py --split $i \
                                   --ds_name "PNLPhub/digikala-sentiment-analysis,PNLPhub/Pars-ABSA,PNLPhub/PEYMA,PNLPhub/C-ExaPPC,PNLPhub/DigiMag,parsinlu_reading_comprehension,PNLPhub/Persian-News" \
                                   --generate_metadata \
                                   --shots 3


done
cd ..
