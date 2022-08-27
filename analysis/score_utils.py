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


def add_max_score_to_df(df):
    """ Adds the highest score of each subject to the given dataframe """

    scores = read_score_data()

    # only for last level
    highest_level = scores['after_levels'].max()
    highest_level_mask = scores['after_levels'] == highest_level

    highest_level_scores = scores[highest_level_mask]

    # merge with df
    return df.merge(highest_level_scores[['subject_id', 'score']], on='subject_id', how='left')
