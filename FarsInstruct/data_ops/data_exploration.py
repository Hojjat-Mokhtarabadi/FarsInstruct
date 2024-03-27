import pandas as pd

def gen_data():
    df = pd.read_csv('data/p3_test.csv')
    df['source'] = 'p3'
    # print(df.head())
    df2 = pd.read_csv('data/1shot_instruct_dataset_test.csv')
    df2['source'] = 'farsinstruct'

    df3 = pd.concat([df, df2])

    df3.to_csv('data/1shot_farsintruct_p3_test.csv', index=False, header=True)

def read_data():
    df = pd.read_csv('data/mixed_instruct_dataset_train.csv')
    # print(df.head())
    print(df['ds'].value_counts())
    print(len(df['ds'].unique()))
    print(len(df))

def get_zs_sample():
    df = pd.read_csv('data/1shot_farsintruct_p3_train.csv')
    all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'ds', 'template'])
    for i in df['ds'].unique():
        min_chunk = 11_000
        dff = df[df['ds'] == i]
        spl = dff.sample(min(min_chunk, len(dff))).reset_index()
        all_dfs = pd.concat([all_dfs, spl])
        # print(spl['inputs'][0])
        # print(10 * '*')
        # print(spl['outputs'][0])
        # print(50 * '-')
        # txt += (spl['inputs'][0] + '\n' + 10*'*' + '\n' + spl['outputs'][0] + '\n' + 50*'-' + '\n')

    all_dfs.to_csv('data/sample_of_zs_data.csv', index=False, header=True)
    # with open('data/all_3shot_data_sample.txt', 'w') as f:
    #     f.write(txt)

def get_fs_sample():
    df = pd.read_csv('data/3shot_instruct_dataset_train.csv')
    all_dfs = pd.DataFrame(columns=['inputs', 'outputs', 'ds', 'template'])
    for i in df['ds'].unique():
        min_chunk = 15_000
        dff = df[df['ds'] == i]
        spl = dff.sample(min(min_chunk, len(dff))).reset_index()
        all_dfs = pd.concat([all_dfs, spl])
        # print(spl['inputs'][0])
        # print(10 * '*')
        # print(spl['outputs'][0])
        # print(50 * '-')
        # txt += (spl['inputs'][0] + '\n' + 10*'*' + '\n' + spl['outputs'][0] + '\n' + 50*'-' + '\n')

    all_dfs.to_csv('data/sample_of_fs_data.csv', index=False, header=True)

def read_sample_data():
    df1 = pd.read_csv('data/sample_of_zs_data.csv')
    df2 = pd.read_csv('data/sample_of_fs_data.csv')

    print(df1['ds'].value_counts())
    print(df2['ds'].value_counts())
    print(len(df1))
    print(len(df2))


def conb(): 
    df1 = pd.read_csv('data/sample_of_zs_data.csv')
    df2 = pd.read_csv('data/sample_of_fs_data.csv')

    df = pd.concat([df1, df2])
    df.to_csv('data/mixed_instruct_dataset_train.csv', index=False, header=True)

def prt():
    df1 = pd.read_csv('data/mixed_instruct_dataset_train.csv')
    df2 = pd.read_csv('data/1shot_instruct_dataset_validation.csv')

    print(df1.head())
    print(df2.head())

def fix():
    df1 = pd.read_csv('data/mixed_instruct_dataset_train.csv')
    df1 = df1.drop(columns=['index', 'source'])

    df1.to_csv('data/mixed_instruction_dataset_train.csv', index=False, header=True)



read_data()
# fix()
