# FarsInstruct

**[2025.01.20]** 🏆 Our paper was nominated as the best paper at <a href="https://loreslm.github.io/">LowResLM @ COLING 2025!</a> <br>
**[2024.12.07]** ✨ Our paper has been accepted for oral presentation at <a href="https://loreslm.github.io/">LowResLM @ COLING 2025!</a>

This repository contains the complete implementation of our research paper "*Empowering Persian LLMs for Instruction Following: A Novel Dataset and Training Approach*" presented at the First Workshop on Language Models for Low-Resource Languages (LoResLM 2025).

## Key Features
**Comprehensive Dataset**  
197 carefully crafted prompt templates across 21 distinct Persian NLP datasets

**Novel Training Framework**  
Co-CoLA (Continual-Chain of LoRA) for enhanced multi-task adaptability

**Diverse Task Coverage**  
10 different task types including NER, sentiment analysis, summarization, and more

**Cultural Authenticity**  
Human-annotated instructions ensuring linguistic and cultural relevance

**Open Source**  
Fully available for research and development [🤗 Hugging Face](https://huggingface.co/datasets/PNLPhub/FarsInstruct)

## 🚀 Quick Start

### Requirements:
- python 3.9
- 16GB+ RAM

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hojjat-Mokhtarabadi/FarsInstruct.git
   cd FarsInstruct
   ```

2. **Set up the environment**:
   ```bash
   bash _setup.sh
   ```

3. **Generate the dataset**:
   ```bash
   bash _build_data.sh
   ```

### Usage

#### Training

```bash
bash _run_fine_tune.sh
```

**Note**: Set `--dataload_mode` to either `'local'` or `'hub'` depending on your data source preference.

#### Evaluation

```bash
bash _run_evaluation.sh
```

## Citation

If you use FarsInstruct in your research, please cite our paper:

```bibtex
@inproceedings{mokhtarabadi-etal-2025-empowering,
    title = "Empowering {P}ersian {LLM}s for Instruction Following: A Novel Dataset and Training Approach",
    author = "Mokhtarabadi, Hojjat  and
      Zamani, Ziba  and
      Maazallahi, Abbas  and
      Manshaei, Mohammad Hossein",
    editor = "Hettiarachchi, Hansi  and
      Ranasinghe, Tharindu  and
      Rayson, Paul  and
      Mitkov, Ruslan  and
      Gaber, Mohamed  and
      Premasiri, Damith  and
      Tan, Fiona Anting  and
      Uyangodage, Lasitha",
    booktitle = "Proceedings of the First Workshop on Language Models for Low-Resource Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.loreslm-1.3/",
    pages = "31--67",
    abstract = "Instruction-tuned large language models have demonstrated remarkable capabilities in following human instructions across various domains. However, their proficiency remains notably deficient in many low-resource languages. To address this challenge, we begin by introducing FarsInstruct: a comprehensive instruction dataset designed to enhance the instruction-following ability of large language models specifically for the Persian language{---}a significant yet underrepresented language globally. FarsInstruct encompasses a wide range of task types and datasets, each containing a mix of straightforward to complex manual written instructions, as well as translations from the Public Pool of Prompts, ensuring a rich linguistic and cultural representation. Furthermore, we introduce Co-CoLA, a framework designed to enhance the multi-task adaptability of LoRA-tuned models. Through extensive experimental analyses, our study showcases the effectiveness of the FarsInstruct dataset coupled with training by the Co-CoLA framework, in improving the performance of large language models within the Persian context. As of the current writing, FarsInstruct comprises 197 templates across 21 distinct datasets, and we intend to update it consistently, thus augmenting its applicability."
}
```