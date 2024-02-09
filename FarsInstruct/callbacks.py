from transformers.integrations import WandbCallback
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM

import torch
import wandb
import tqdm

from FarsInstruct.evaluation.run_eval import LMEvaluation


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, fars_config):
        "A CallBack to log samples a wandb.Table during training"
        super().__init__()
        self.trainer = trainer
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.lm_eval = LMEvaluation(fars_config, self.tokenizer, split='validation')
        self.fars_config = fars_config
        
    def on_log(self, args, state, control,  **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_log(args, state, control, **kwargs)

        print("\n#### Running Evaluation... ####")
        all_results,samples = self.lm_eval.run_eval(current_model=self.model, step=self.trainer.state.global_step)
        #records_table = self.samples_table(self.sample_dataset)
        #self._wandb.log({"evaluation_results":all_results['Evaluation results']})
        for x in all_results:
            self._wandb.log({x['ds_name']:{x['temp_name']:x['result']}})
        for x in samples:
            if len(x['tokens'])>2:
                self._wandb.log({x['ds_name']:{x['temp_name']:f"{x['tokens'][0]} => {x['tokens'][2]}"}})
        #self._wandb.log({"generate_samples":samples})

        #print("\n#### Sampling... ####")
        #sample = self.lm_eval.generate(self.fars_config)
        #self._wandb.log({"sample":sample})
