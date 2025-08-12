git clone --branch persian_temps https://github.com/Hojjat-Mokhtarabadi/promptsource.git
cd promptsource
pip install -e .
cd ..

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..


pip install -e .
