from pathlib import Path
import yaml
from jinja2 import Template

script_path = Path(__file__).parent
        
class Prompter:
    def __init__(self, model_family):
        self.model_family = model_family
        self.model_temp = self.load_prompt_temp()
        
    def load_prompt_temp(self):
        with open(script_path / "templates.yaml", "r") as f:
            model_temp = yaml.safe_load(f)[self.model_family]
            
        return model_temp

    def render(self, msg: str, prompt_format: str = "sft", render_for_dataset_map: bool = True, **kwargs):
        if prompt_format == "pretraining":
            template = Template(self.model_temp[0]['pretrain_prompt'])
            rendered = template.render(user_message=msg)

        elif prompt_format == "sft":
            template = Template(self.model_temp[1]['instruct_prompt'])
            rendered = template.render(user_message=msg)
            
        elif prompt_format == "multi_choice_prompt":
            template = Template(self.model_temp[2]["multi_choice_prompt"])
            rendered = template.render(user_message=msg, op1=kwargs['op1'], op2=kwargs['op2'], op3=kwargs['op3'], op4=kwargs['op4'])
            
        elif phase == "none":
            rendered = msg

        return rendered
    
    def get_response(self, model_output: str, phase: str):
        
        return 
        
        
