cd FarsInstruct

#accelerate launch peft_trainer.py --dataload_mode local
accelerate launch vanilla_trainer.py --dataload_mode local
cd ..
