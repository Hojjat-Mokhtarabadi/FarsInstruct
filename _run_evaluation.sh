python lm-evaluation-harness/main.py --model hf \
                                     --model_args pretrained=HooshvareLab/gpt2-fa \
                                     --device cuda \
                                     --output_path ./evaluation_results \
                                     --tasks can_you_infer_zs_promptsource,summarize_the_article_zs_promptsource \
                                     --num_fewshot 0