import pandas as pd

from analysis.data_utils import read_data


def load_sosci_csv():
    SOSCI_PATH = '../data/sosci_data.csv'
    df = pd.read_csv(SOSCI_PATH, delimiter='\t', encoding='utf-16')
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # make ids uppercase
    df['subject_id'] = df['subject_id'].apply(lambda x: x.upper())

    # one subject noted wrong code in survey
    df.loc[df['subject_id'] == 'JO10SA'] = 'JO03SA'
    return df


def add_experience_to_df(df):
    experience_df = load_sosci_csv()
    df = df.merge(experience_df, on='subject_id', how='left')
    return df


# df = read_data()
# modified_df = add_experience_to_df(df)
# test = modified_df.groupby(['subject_id']).head(1)
# test2 = modified_df[modified_df['subject_id'] == 'ZI01SU']
# print(test2)
