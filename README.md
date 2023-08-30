# FarsInstruct
Instruction tuning of Persian LLMs

### Requirements:
- python 3.8

### Steps:
First install the package, using:
```bash
pip install -e .
```

In order to build the instruction based dataset from scratch:
```bash
bash build_data.sh
```
Start training:

NOTE: --dataload_mode should be either 'local' or 'hub'
```bash
accelerate launch vanilla_trainer.py --dataload_mode hub
```


## TODOs
- [x] Push instruction dataset to huggingface hub
- [x] Fix digikala-sentiment-analysis templates for FS instructions
- [ ] Document the code
- [x] Implement data streaming + data loading 
- [x] Setup standard fine-tuning procedure with Accelerate
- [ ] Check Parameter efficient fine-tuning methods, e.g. LoRA, QLoRA, Prompt Tuning
- [ ] Check T5 model as the base
