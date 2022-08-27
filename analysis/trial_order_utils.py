import os

import pandas as pd

from analysis.data_utils import get_all_subjects


def load_trial_order_csv(path):
    return pd.read_csv(path, sep=';', names=['trial', 'difficulty', 'world_name'])


def load_trial_orders():
    """ Loads all trial_nr information for all subjects """
    TRIAL_ORDER_PATH_GENERAL = '../data/level_data/{}/trial_order.csv'

    trial_order_dfs = []

    for subject_id in get_all_subjects():
        TRIAL_ORDER_PATH_SUBJECT = TRIAL_ORDER_PATH_GENERAL.format(subject_id)

        if not os.path.exists(TRIAL_ORDER_PATH_SUBJECT):
            continue

        trial_order_df = load_trial_order_csv(TRIAL_ORDER_PATH_SUBJECT)

        # get world number from world name and delete world_name column
        trial_order_df['world_number'] = trial_order_df['world_name'].apply(lambda x: x.split('_')[-1])
        trial_order_df.drop('world_name', axis=1, inplace=True)

        # add subject_id
        trial_order_df['subject_id'] = subject_id

        trial_order_dfs.append(trial_order_df)

    return pd.concat(trial_order_dfs)



