from setuptools import setup, find_packages

requirements = [
        # 'transformers==4.33.1',
        'transformers',
        'hazm',
        'pandas',
        'trl',
        'scikit-learn',
        'bitsandbytes',
        'peft',
        'evaluate',
        'prettytable'
]

setup(
    name='FarsInstruct',
    version='0.1',
    author='Hojjat Mokhtarabadi',
    python_requires=">3.9.*",
    install_requires=requirements,
    packages=find_packages(exclude=['promptsource', 'results', 'wandb', 'checkpoints'])
)
