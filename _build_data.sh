cd FarsInstruct
git clone --branch persian_temps https://github.com/Hojjat-Mokhtarabadi/promptsource.git
cd promptsource
pip install -e .
cd ..

python build_data_gym/build_gym.py --split validation