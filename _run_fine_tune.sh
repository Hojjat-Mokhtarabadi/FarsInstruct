cd FarsInstruct

accelerate launch peft_trainer.py --dataload_mode hub
# accelerate launch vanilla_trainer.py --dataload_mode local
# accelerate launch stf_trainer.py

cd ..
