from datasets import load_dataset
import pandas as pd

ds1 = load_dataset("PNLPhub/P3-QA-translated")
ds2 = load_dataset("PNLPhub/P3-XL-WiC")

df1 = ds1['test'].to_pandas()
df2 = ds2['test'].to_pandas().rename(columns={"label":'outputs', 'temp_name': 'template'})

df = pd.concat([df1, df2])

df.to_csv('p3_test.csv', index=False, header=True)