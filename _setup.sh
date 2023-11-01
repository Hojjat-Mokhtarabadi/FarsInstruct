pip3 install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple


git clone --branch persian_temps https://github.com/Hojjat-Mokhtarabadi/promptsource.git
cd promptsource
pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

git clone --branch big-refactor-fa https://github.com/Hojjat-Mokhtarabadi/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
cd .. 

pip install -e .
