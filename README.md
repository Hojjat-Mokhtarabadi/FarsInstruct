# FarsInstruct
Instruction tuning of Persian LLMs

### Requirements:
- python 3.9

### Steps:
Run the following scripts:

Initial setup:
```bash
bash _setup.sh
```
Data generation:
```bash
bash _build_data.sh
```
Start training:

NOTE: --dataload_mode should be either 'local' or 'hub'
```bash
bash _run_fine_tune.sh
```

Evaluation:
```bash
bash _run_evaluation.sh
```


## TODOs
- [x] Push instruction dataset to huggingface hub
- [x] Fix digikala-sentiment-analysis templates for FS instructions
- [ ] Document the code
- [x] Implement data streaming + data loading 
- [x] Setup standard fine-tuning procedure with Accelerate
- [ ] Check Parameter efficient fine-tuning methods, e.g. LoRA, QLoRA, Prompt Tuning
- [ ] Check T5 model as the base
