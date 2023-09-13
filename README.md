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
## Some details
- Num of prompted datasets: 14
- Num of prompts: 72
- Base-model: HooshvareLab/gpt2-fa
- Num params: ~118M

## Prompted tasks
- [x] parsinlu_reading_comprehension
- [x] persiannlp/parsinlu_entailment
- [x] persiannlp/parsinlu_query_paraphrasing
- [x] persiannlp/parsinlu_sentiment
- [x] PNLPhub/digikala-sentiment-analysis
- [x] PNLPhub/DigiMag
- [x] PNLPhub/FarsTail
- [x] PNLPhub/parsinlu-multiple-choice
- [x] PNLPhub/Persian-News
- [x] PNLPhub/snappfood-sentiment-analysis
- [x] pn_summary
- [x] SajjadAyoubi/persian_qa
- [x] SLPL/syntran-fa
- [x] wiki_summary
- [ ] PEYMA
- [ ] Persian NER

## Available groups on lm-eval
- farstail
- parsinlu_entailment
- parsinlu-multiple-choice
- parsinlu_sentiment
- parsinlu_paraphrase
- parsinlu_reading_comprehension
- digi-sentiment
- snapp_sentiment
- persian_qa

## Scoring details:
- **output_type** (`str`, *optional*, defaults to "greedy_until") — Selects the type of model output for the given task. Options are `greedy_until`, `loglikelihood`, `loglikelihood_rolling`, and `multiple_choice`.
- **supported metrics** -  `acc` (accuracy), `acc_norm` (length-normalized accuracy), `acc_mutual_info` (baseline loglikelihood - normalized accuracy), `perplexity`, `word_perplexity` (perplexity per word), `byte_perplexity` (perplexity per byte), `bits_per_byte`, `matthews_corrcoef` (Matthews correlation coefficient), `f1` (F1 score), `bleu`, `chrf`, `ter`

## Good Reference Tasks

Contributing a new task can be daunting! Luckily, much of the work has often been done for you in a different, similarly evaluated task. Good examples of task implementations to study include:

Multiple choice tasks:
- SciQ (`lm_eval/tasks/sciq/sciq.yaml`)

Corpus perplexity evaluations:
- Wikitext (`lm_eval/tasks/wikitext/wikitext.yaml`)

Generative tasks:
- GSM8k (`lm_eval/tasks/gsm8k/gsm8k.yaml`)

Tasks using complex filtering:
- GSM8k with CoT (+ with Self-Consistency): (`lm_eval/tasks/gsm8k/gsm8k-cot.yaml` ; `lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml`)
