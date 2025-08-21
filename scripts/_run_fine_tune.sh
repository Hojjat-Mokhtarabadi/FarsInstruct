cd FarsInstruct

accelerate launch peft_trainer.py
# accelerate launch vanilla_trainer.py --dataload_mode local
# accelerate launch stf_trainer.py

cd ..
