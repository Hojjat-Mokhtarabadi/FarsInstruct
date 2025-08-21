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
    
    def get_response_key(self):
        if self.model_family == "llama3x":
            return "<|start_header_id|>assistant<|end_header_id|>"
        if self.model_family == "qwen3":
            return "<|im_start|>assistant\n"

    def render(self, msg: str, prompt_format: str = "sft", **kwargs):
        if prompt_format == "pretrain":
            template = Template(self.model_temp[0]['pretrain'])
            rendered = template.render(user_message=msg)

        elif prompt_format == "sft":
            template = Template(self.model_temp[1]['instruct'])
            rendered = template.render(user_message=msg)
            
        elif prompt_format == "multi_choice":
            template = Template(self.model_temp[2]["multi_choice"])
            rendered = template.render(user_message=msg, op1=kwargs['op1'], op2=kwargs['op2'], op3=kwargs['op3'], op4=kwargs['op4'])
            
        elif phase == "none":
            rendered = msg

        return rendered
    
    def get_response(self, model_output: str, phase: str):
        
        return 
        
        
