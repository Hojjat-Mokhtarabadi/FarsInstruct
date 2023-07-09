pip install -r requirements.txt

git clone https://github.com/Hojjat-Mokhtarabadi/promptsource.git
cd promptsource
pip install -e .
cd ..

python build_data_gym/build_gym.py

python data/hf_dataset.py