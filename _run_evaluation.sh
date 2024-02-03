cd FarsInstruct

# accelerate launch evaluation/run_eval.py --task_type multiple_choice
python evaluation/run_eval.py --split test

cd ..