from FarsInstruct.evaluation.run_eval import LMEvaluation
from transformers.integrations import TensorBoardCallback
import time
from torch.utils.tensorboard import SummaryWriter

            
class LLMTensorboardCallback(TensorBoardCallback):
    def __init__(self, trainer, fars_config, logging_dir, run_name):
        "A CallBack to log samples a wandb.Table during training"
        super().__init__()
        self.trainer = trainer
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.lm_eval = LMEvaluation(fars_config, self.tokenizer, split='validation')
        self.fars_config = fars_config
        self.logging_dir = logging_dir
        self.run_name = f'{run_name}/{str(time.time())}'
        self.writer = SummaryWriter(log_dir=f'{self.logging_dir}/{self.run_name}')
        
    def on_log(self, args, state, control,  **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_log(args, state, control, **kwargs)

        print("\n#### Running Evaluation... ####")

        try:
            all_results, samples = self.lm_eval.run_eval(current_model=self.model, step=self.trainer.state.global_step)
            
            self.writer = SummaryWriter(log_dir=f'{self.logging_dir}/{self.run_name}')
            for x in all_results:
                for key in x['result']:
                    self.writer.add_scalar(f"logs/evaluating/{x['ds_name']}/{x['temp_name']}/{key}",x['result'][key],self.trainer.state.global_step)
            for sample in samples:
                for i, token in enumerate(['input','preds','label']):
                    self.writer.add_text(f"logs/evaluating/{sample['ds_name']}/{sample['temp_name']}/{token}",str(sample['tokens'][i][0]),self.trainer.state.global_step)
        except Exception as e:
            print(e)



