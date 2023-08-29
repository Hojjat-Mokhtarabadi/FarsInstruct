from setuptools import setup, find_packages

requirements = [
        'torch', 
        'hazm',
        'transformers',
        'datasets',
        'accelerate',
        'pandas',
        'numpy<1.24',
        'bitsandbytes',
        'trl',
        'peft @ git+https://github.com/huggingface/peft.git',
        'wandb',
        'scipy',
        'scikit-learn'
]

setup(
    name='FarsInstruct',
    version='0.1',
    author='Hojjat Mokhtarabadi',
    install_requires=requirements,
    packages=find_packages(exclude=['promptsource', 'results', 'wandb', 'checkpoints'])
)