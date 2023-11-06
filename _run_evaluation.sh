python lm-evaluation-harness/main.py --model hf \
                                     --model_args pretrained=./FarsInstruct/checkpoints/llama2.zs-fs.digi.-1.bs1\
                                     --device cuda \
                                     --output_path ./evaluation_results/digi_snapp_on_sentiment_llama.json \
                                     --tasks PNLPhub/digikala-sentiment-analysis, PNLPhub/FarsTail, PNLPhub/parsinlu-multiple-choice, PNLPhub/snappfood-sentiment-analysis  \
                                     --num_fewshot 0