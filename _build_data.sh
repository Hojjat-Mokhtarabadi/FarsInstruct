cd FarsInstruct

python build_data_gym/build_gym.py --split train --generate_metadata \
                                   --ds_name PNLPhub/snappfood-sentiment-analysis,PNLPhub/digikala-sentiment-analysis,pn_summary

cd ..
