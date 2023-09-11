pip3 install torch --index-url https://download.pytorch.org/whl/cu117

git clone --branch persian_temps https://github.com/Hojjat-Mokhtarabadi/promptsource.git
cd promptsource
pip install -e . --no-cache-dir
cd ..

git clone --branch big-refactor-fa https://github.com/Hojjat-Mokhtarabadi/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e . --no-cache-dir
cd .. 

pip install -e .
