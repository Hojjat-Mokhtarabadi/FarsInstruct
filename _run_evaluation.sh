python lm-evaluation-harness/main.py --model hf \
                                     --model_args pretrained=./FarsInstruct/checkpoints/hooshvare.zs-fs.digi-snapp-pn-sum.-1.bs8\
                                     --device cuda \
                                     --output_path ./evaluation_results/digi_snapp_on_sentiment_1.json \
                                     --tasks PNLPhub/digikala-sentiment-analysis, PNLPhub/FarsTail, PNLPhub/parsinlu-multiple-choice, PNLPhub/snappfood-sentiment-analysis  \
                                     --num_fewshot 0