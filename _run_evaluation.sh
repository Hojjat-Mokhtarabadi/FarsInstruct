python lm-evaluation-harness/main.py --model hf \
                                     --model_args pretrained=HooshvareLab/gpt2-fa \
                                     --device cuda \
                                     --output_path ./evaluation_results \
                                     --tasks parsinlu-multiple-choice \
                                     --num_fewshot 0