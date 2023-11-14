# python lm-evaluation-harness/main.py --model hf \
#                                      --model_args pretrained=./FarsInstruct/checkpoints/hooshvare.zs-fs.digi-snapp.1.bs1\
#                                      --device cuda \
#                                      --output_path ./evaluation_results/digi_snapp_on_sentiment_llama2.json \
#                                      --tasks farstail-llama \
#                                      --limit 1 \
#                                      --num_fewshot 0

python lm-evaluation-harness/main.py --model hf \
                                     --model_args pretrained="F:\Pretrained_models\Llama2-7b-hf-raw",peft="C:\Users\Hojjat\Python projects\llama2.zs-fs.digi.-1.bs1_pretrained"\
                                     --device cuda \
                                     --output_path ./evaluation_results/digi_snapp_on_sentiment_llama3.json \
                                     --tasks farstail-llama \
                                     --limit 1 \
                                     --num_fewshot 0