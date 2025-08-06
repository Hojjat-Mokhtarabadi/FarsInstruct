import os
import requests
import zipfile
from pathlib import Path


class SNIDatasetBuilder:
    def __init__(self, cache_dir):
        self.tasks_path = None
        self.cache_dir: str = cache_dir
        self.task_data = {}

    def download_data(self, force_download: bool):
        
        # handle cache dir
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_path = os.path.join(self.cache_dir, "natural-instructions")
        
        if os.path.exists(dataset_path) and not force_download:
            print(f"Dataset already exists at {dataset_path}")
            self.tasks_dir = os.path.join(dataset_path, "tasks")
            self._load_all_tasks()
            retur
        
        print("Downloading sni data!")
        zip_path = os.path.join(self.cache_dir, "natural-instructions.zip")

        # download repository as a zip file
        url = ""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
                
        # extract zip file
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extactall(self.cache_dir)
            
        # rename the extracted folder
        extracted_folder = os.path.join(self.cache_dir, "natural-instructions-master")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.rename(extracted_folder, dataset_path)
        
        # clean up zip file
        os.remove(zip_path)
        
        self.task_dir = os.path.join(dataset_path, "tasks")
        self._load_all_tasks()
        print(f"Downloaded dataset to {dataset_path}")
        
    
    def _load_all_tasks(self):
        """Load all task JSON files into memory."""
        if not self.tasks_dir or not os.path.exists(self.tasks_dir):
            raise ValueError("Tasks directory not found. Please download the dataset first.")

        print("Loading all tasks...")
        task_files = [f for f in os.listdir(self.tasks_dir) if f.endswith('.json')]

        for task_file in tqdm(task_files, desc="Loading tasks"):
            task_name = task_file.replace('.json', '')
            task_path = os.path.join(self.tasks_dir, task_file)

            try:
                with open(task_path, 'r', encoding='utf-8') as f:
                    self.task_data[task_name] = json.load(f)
            except Exception as e:
                print(f"Error loading {task_file}: {e}")

            
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
    
    
    def discover_language_tasks(self, language: str) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Automatically discover all tasks that support a given language and group by category.

        Args:
            language: Language to discover tasks for

        Returns:
            Dictionary mapping categories to list of (task_name, task_data) tuples
        """
        category_tasks = defaultdict(list)

        for task_name, task_data in self.task_data.items():
            # Check if language is in input or output languages
            input_langs = task_data.get('Input_language', [])
            output_langs = task_data.get('Output_language', [])

            # Handle both string and list formats
            if isinstance(input_langs, str):
                input_langs = [input_langs]
            if isinstance(output_langs, str):
                output_langs = [output_langs]

            # Check if the language is supported
            if language in input_langs or language in output_langs:
                # Get categories from the task
                categories = task_data.get('Categories', [])
                if isinstance(categories, str):
                    categories = [categories]

                # If no categories specified, use a default category
                if not categories:
                    categories = ['Uncategorized']

                # Add task to all its categories
                for category in categories:
                    category_tasks[category].append((task_name, task_data))

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
        if not self.task_data:
            raise ValueError("No tasks loaded. Please download the dataset first.")

        # Discover all tasks for the language
        language_tasks = self.discover_language_tasks(language)
        
        if not language_tasks:
            raise ValueError(f"No tasks found for language: {language}")

        # Filter categories if specified
        if categories:
            language_tasks = {cat: tasks for cat, tasks in language_tasks.items() if cat in categories}

        # Build dataset entries
        dataset_entries = {
            'input': [],
            'target': [],
            'task_name': [],
            'category_name': []
        }

        print(f"\nBuilding dataset for language: {language}")
        print(f"Found {len(language_tasks)} categories with tasks")
        
        for category, task_list in tqdm(language_tasks.items(), desc="Processing categories"):
            for task_name, task_data in task_list:
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

                    dataset_entries['input'].append(instruction)
                    dataset_entries['target'].append(target)
                    dataset_entries['task_name'].append(task_name)
                    dataset_entries['category_name'].append(category)
                    
        # Create HuggingFace dataset
        dataset = Dataset.from_dict(dataset_entries)

        print(f"\nDataset created successfully!")
        print(f"Total examples: {len(dataset)}")
        print(f"Categories included: {sorted(language_tasks.keys())}")
        print(f"Total unique tasks: {len(set(dataset['task_name']))}")

        return dataset
    
if __name__ == "__main__":
    CACHE_DIR = "wrkdir/Hojjat_Workstation/instruction_tuning/FarsInstruct/FarsInstruct/data/sni_data"
    TASKS_OUTPUT_PATH = Path(CACHE_DIR) / "persian_tasks_full.json"
    LANG = "Persian"
    
    sni_builder = SNIDatasetBuilder(CACHE_DIR)
    sni_builder.download_dataset()
    sni_dataset.save_language_tasks_to_json(
        LANG,
        output_path=TASKS_OUTPUT_PATH,
        save_full_tasks=False
    )
    ds = sni_dataset.build_dataset(LANG)

    