# FarsInstruct
Instruction tuning of Persian LLMs

### Requirements:
- python 3.8

### Steps:
In order to build the instruction based dataset from scratch:
```bash
bash setup.sh
bash build_data.sh
```
Start training:

NOTE: --dataload_mode should be either 'local' or 'hub'
```bash
accelerate launch main.py --dataload_mode hub
```


## TODOs
- [x] Push instruction dataset to huggingface hub
- [] Fix digikala-sentiment-analysis templates for FS instructions
- [] Document the code
- [x] Implement data streaming + data loading 
- [x] Setup standard fine-tuning procedure with Accelerate
- [] Check Parameter efficient fine-tuning methods, e.g. LoRA, QLoRA, Prompt Tuning
- [] Implement evaluation setup, i.e. val-set, metrics
