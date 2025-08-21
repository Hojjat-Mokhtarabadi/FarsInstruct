import os
import requests
import zipfile
import shutil
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Generator
from torch.utils.data import Dataset as DS
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
from data_ops.prompter import Prompter


CACHE_DIR = "/mnt/beegfs/wrkdir/u111187/Hojjat_Workstation/instruction_tuning/FarsInstruct/data/sni_data"

class SNIDataset:
    def __init__(self, tokenizer: AutoTokenizer, max_len: int, max_task_examples: int, lang: str, model_family: str, 
                             categories: Optional[List[str]] = None, cache_dir: str = CACHE_DIR):
        self.tasks_dir = None
        self.cache_dir: str = cache_dir
        self.task_files = []  # Just store filenames
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_task_examples = max_task_examples
        
        self.prompter = Prompter(model_family)
        
        # load dataset if not available and load all tasks
        self.download_data()
        self.ds = self.build_dataset(lang, categories)

        
    def get_tokenized_data(self, in_torch_format: bool):
        self.tokenized_ds = self.ds.map(self.tokenize_fn, batched=True, desc="tokenizing data...")
        
        if in_torch_format:
            return self.tokenized_ds.with_format("torch", columns=["input_ids", "attention_mask"])
        else:
            return self.tokenized_ds
        

    def tokenize_fn(self, examples):
        formatted_exp = []
        inputs = examples["input"]
        outputs = examples["target"]
                
        for i , o in zip(inputs, outputs):
            input_ = self.prompter.render(i, prompt_format="sft")
            output_ = o
            full_prompt = input_ + output_
            formatted_exp.append(full_prompt)
                         
        tokenized = self.tokenizer(
            formatted_exp,
            padding="max_length", 
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=False
        )   
        
        del formatted_exp
        
        return tokenized
        
        
    def download_data(self, force_download: bool = False):
        # handle cache dir
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_path = os.path.join(self.cache_dir, "natural-instructions")

        if os.path.exists(dataset_path) and not force_download:
            print(f"Dataset already exists at {dataset_path}")
            self.tasks_dir = os.path.join(dataset_path, "tasks")
            self._load_all_tasks()
            return

        print("Downloading sni data...")
        zip_path = os.path.join(self.cache_dir, "natural-instructions.zip")

        # download repository as a zip file
        url = "https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)

        # extract zip file
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)

        # rename the extracted folder
        extracted_folder = os.path.join(self.cache_dir, "natural-instructions-master")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.rename(extracted_folder, dataset_path)

        # clean up zip file
        os.remove(zip_path)

        self.tasks_dir = os.path.join(dataset_path, "tasks")
        self._load_all_tasks()
        print(f"Downloaded dataset to {dataset_path}")

        
    def _load_all_tasks(self):
        """Load task JSON files into memory."""
        if not self.tasks_dir or not os.path.exists(self.tasks_dir):
            raise ValueError("Tasks directory not found. Please download the dataset first.")

        print("Getting task file list...")
        self.task_files = [f for f in os.listdir(self.tasks_dir) if f.endswith('.json')]
        print(f"Found {len(self.task_files)} task files")

        
    def _load_single_task(self, task_file: str) -> Dict:
        """Load a single task file on demand."""
        task_path = os.path.join(self.tasks_dir, task_file)
        try:
            with open(task_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {task_file}: {e}")
            return {}

            
    def _create_instruction(self, task_data: Dict, instance: Dict) -> str:
        """
        Create instruction following the paper's format.
        Based on the paper, definition + 2 positive examples works best.

        Args:
            task_data: The task dictionary
            instance: The specific instance

        Returns:
            Formatted instruction string
        """
        instruction_parts = []

        # Add definition
        definitions = task_data.get('Definition', [])
        if definitions:
            # Handle both string and list formats
            definition = definitions[0] if isinstance(definitions, list) else definitions
            instruction_parts.append(f"Definition: {definition}")

        # Add positive examples (up to 2, without explanations based on paper findings)
        positive_examples = task_data.get('Positive Examples', [])
        for i, example in enumerate(positive_examples[:2], 1):
            instruction_parts.append(f"\nPositive Example {i}-")
            instruction_parts.append(f"input: {example.get('input', '')}")
            instruction_parts.append(f"output: {example.get('output', '')}")
            # Note: Paper shows explanations are not helpful for smaller models

        # Add the instance input
        instruction_parts.append("\nNow complete the following example-")
        instruction_parts.append(f"input: {instance.get('input', '')}")
        instruction_parts.append("output:")

        return "\n".join(instruction_parts)
    
    
    def discover_language_tasks(self, language: str) -> Dict[str, List[str]]:
        """
        Discover task files that support a language without loading all data.
        Returns category -> list of task filenames mapping.
        """
        category_tasks = defaultdict(list)
        
        print(f"Discovering tasks for language: {language}...")
        
        for task_file in tqdm(self.task_files, desc="Scanning tasks"):
            # Load task data on demand
            task_data = self._load_single_task(task_file)
            if not task_data:
                continue
                
            # Check if language is supported
            input_langs = task_data.get('Input_language', [])
            output_langs = task_data.get('Output_language', [])

            if isinstance(input_langs, str):
                input_langs = [input_langs]
            if isinstance(output_langs, str):
                output_langs = [output_langs]

            if language in input_langs or language in output_langs:
                categories = task_data.get('Categories', [])
                if isinstance(categories, str):
                    categories = [categories]
                if not categories:
                    categories = ['Uncategorized']

                for category in categories:
                    category_tasks[category].append(task_file)
        
        return dict(category_tasks)
    
    
    def save_language_tasks_to_json(self, language: str, output_path: Optional[str] = None,
                                   save_full_tasks: bool = False) -> str:
        """
        Save discovered tasks for a language to a JSON file.

        Args:
            language: Language to save tasks for
            output_path: Optional custom output path. If None, saves to cache directory
            save_full_tasks: If True, saves complete task data. If False, saves only task names.

        Returns:
            Path to saved JSON file
        """
        # Discover tasks for the language
        language_tasks = self.discover_language_tasks(language)

        if not language_tasks:
            raise ValueError(f"No tasks found for language: {language}")

        # Prepare output path
        if output_path is None:
            output_dir = os.path.join(self.cache_dir, "language_tasks")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{language.lower()}_tasks.json")

        # Prepare data to save
        if save_full_tasks:
            # Save complete task data
            save_data = {}
            for category, task_list in language_tasks.items():
                save_data[category] = []
                for task_name, task_data in task_list:
                    save_data[category].append({
                        'task_name': task_name,
                        'task_data': task_data
                    })
        else:
            # Save only task names (more compact)
            save_data = {}
            for category, task_list in language_tasks.items():
                save_data[category] = [task_name for task_name, _ in task_list]

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {language} tasks to: {output_path}")
        return output_path

    
    def build_dataset(self, language: str, categories: Optional[List[str]] = None) -> Dataset:
        """
        Build a HuggingFace dataset for the specified language.

        Args:
            language: Language to build dataset for
            categories: Optional list of specific categories to include. If None, includes all categories.

        Returns:
            HuggingFace Dataset with columns: input, target, task_name, category_name
        """
        if not self.task_files:
            raise ValueError("No tasks loaded. Please download the dataset first.")

        # Discover all tasks for the language
        language_tasks = self.discover_language_tasks(language)

        if not language_tasks:
            raise ValueError(f"No tasks found for language: {language}")

        # Filter categories if specified
        if categories:
            language_tasks = {cat: tasks for cat, tasks in language_tasks.items() if cat in categories}

        print(f"\nBuilding dataset for language: {language}")
        print(f"Found {len(language_tasks)} categories with tasks")

        def generate_examples() -> Generator[dict, None, None]:
            total_examples = 0
            outer_break = False
            for category, task_list in tqdm(language_tasks.items(), desc="Processing categories"):
                for task_file in task_list:
                    # Load task data on demand
                    task_data = self._load_single_task(task_file)
                    if not task_data:
                        continue
                        
                    task_name = task_file.replace('.json', '')
                    instances = task_data.get('Instances', [])

                    for instance in instances:
                        # Create instruction-formatted input
                        instruction = self._create_instruction(task_data, instance)

                        # Get target output(s)
                        outputs = instance.get('output', [])
                        if isinstance(outputs, list) and outputs:
                            # Use the first output as target
                            target = outputs[0]
                        else:
                            target = str(outputs) if outputs else ""

                        yield {
                            'input': instruction,
                            'target': target,
                            'task_name': task_name,
                            'category_name': category
                        }
                        total_examples += 1
                        
                        if self.max_task_examples is not None and total_examples >= self.max_task_examples:
                            outer_break = True
                            break
                            
                    if outer_break:
                        break
                        
                    # Clear task_data from memory after processing
                    del task_data
                        
            # These prints happen after generation, but since it's lazy, they'll run post-build
            print(f"\nDataset created successfully!")
            print(f"Total examples: {total_examples}")
            print(f"Categories included: {sorted(language_tasks.keys())}")

        print("Building dataset from generator...")
        dataset = Dataset.from_generator(generate_examples, writer_batch_size=200)

        # Total unique tasks requires loading, but to avoid memory, compute separately if needed
        # For now, approximate or skip; if crucial, use dataset.unique('task_name') after build (low mem)
        print(f"Total unique tasks: (computed post-build if needed)")

        return dataset
    
    
#     def build_dataset(self, language: str, categories: Optional[List[str]] = None) -> Dataset:
#         """
#         Build a HuggingFace dataset for the specified language.

#         Args:
#             language: Language to build dataset for
#             categories: Optional list of specific categories to include. If None, includes all categories.

#         Returns:
#             HuggingFace Dataset with columns: input, target, task_name, category_name
#         """
#         if not self.task_data:
#             raise ValueError("No tasks loaded. Please download the dataset first.")

#         # Discover all tasks for the language
#         language_tasks = self.discover_language_tasks(language)
        
#         if not language_tasks:
#             raise ValueError(f"No tasks found for language: {language}")

#         # Filter categories if specified
#         if categories:
#             language_tasks = {cat: tasks for cat, tasks in language_tasks.items() if cat in categories}

#         # Build dataset entries
#         dataset_entries = {
#             'input': [],
#             'target': [],
#             'task_name': [],
#             'category_name': []
#         }

#         print(f"\nBuilding dataset for language: {language}")
#         print(f"Found {len(language_tasks)} categories with tasks")
        
#         for category, task_list in tqdm(language_tasks.items(), desc="Processing categories"):
#             for task_name, task_data in task_list:
#                 instances = task_data.get('Instances', [])

#                 for instance in instances:
#                     # Create instruction-formatted input
#                     instruction = self._create_instruction(task_data, instance)

#                     # Get target output(s)
#                     outputs = instance.get('output', [])
#                     if isinstance(outputs, list) and outputs:
#                         # Use the first output as target
#                         target = outputs[0]
#                     else:
#                         target = str(outputs) if outputs else ""

#                     dataset_entries['input'].append(instruction)
#                     dataset_entries['target'].append(target)
#                     dataset_entries['task_name'].append(task_name)
#                     dataset_entries['category_name'].append(category)
                    
#         # Create HuggingFace dataset
#         dataset = Dataset.from_dict(dataset_entries)

#         print(f"\nDataset created successfully!")
#         print(f"Total examples: {len(dataset)}")
#         print(f"Categories included: {sorted(language_tasks.keys())}")
#         print(f"Total unique tasks: {len(set(dataset['task_name']))}")

#         return dataset
    
    

if __name__ == "__main__":
    TASKS_OUTPUT_PATH = Path(CACHE_DIR) / "persian_tasks_full.json"
    LANG = "Persian"
    
    sni_builder = SNIDatasetHandler(CACHE_DIR)
    sni_builder.download_data()
    sni_builder.save_language_tasks_to_json(
        LANG,
        output_path=TASKS_OUTPUT_PATH,
        save_full_tasks=False
    )
    ds = sni_builder.build_dataset(LANG)

    