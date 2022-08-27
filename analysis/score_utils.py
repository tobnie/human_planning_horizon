import os

import pandas as pd

from analysis.data_utils import get_all_subjects


def read_level_scores_from_csv(path):
    return pd.read_csv(path, sep=';', names=['subject_id', 'score'])


def filter_score_for_real_data(df):
    """ Only returns the part of the dataframe with real score data."""
    real_subjects = get_all_subjects()
    is_real = df['subject_id'].isin(real_subjects)
    return df[is_real]


def read_score_data():
    SCORE_DATA_PATH = '../data/scores/'

    score_dfs = []
    for f in os.listdir(SCORE_DATA_PATH):
        level_cnt = int(f.split('_')[1])
        level_scores = read_level_scores_from_csv(SCORE_DATA_PATH + f)
        filtered = filter_score_for_real_data(level_scores).copy()
        filtered['after_levels'] = level_cnt
        score_dfs.append(filtered)

    return pd.concat(score_dfs)
