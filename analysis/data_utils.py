import os

import pandas as pd

import config

WIN_THRESHOLD_Y = (config.N_LANES - 2) * config.FIELD_HEIGHT
TIME_OUT_THRESHOLD = config.LEVEL_TIME - 100

SKIP_SUBJECTS = ['MA02CA', 'PE10MI']


def get_all_subjects():
    data_dir = f'../data/level_data/'
    subjects = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir + f) if f not in SKIP_SUBJECTS]
    return subjects


def read_data():
    subject_dfs = []
    for subject in get_all_subjects():
        subject_df = read_subject_data(subject)
        subject_dfs.append(subject_df)
    return pd.concat(subject_dfs)


def read_subject_data(subject_id):
    df = pd.read_csv(f'../data/compressed_data/{subject_id}_compressed.gzip', compression='gzip')
    df.drop(df.columns[0], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df


def drop_missing_samples(df):
    df = df.drop(df[df.gaze_x == -32768].index)
    return df


def get_all_gazes(df=None):
    if not df:
        df = read_data()
    return df.loc[['gaze_x', 'gaze_y']]


def return_status(time, player_y):
    if player_y >= WIN_THRESHOLD_Y:
        return 'won'
    elif time >= TIME_OUT_THRESHOLD:
        return 'timed_out'
    else:
        return 'lost'


def add_game_status_to_df(df):
    last_time_steps = get_last_time_steps_of_games(df).copy()
    last_time_steps['game_status'] = last_time_steps.apply(lambda x: return_status(x['time'], x['player_y']), axis=1)
    df_filtered = last_time_steps[['subject_id', 'game_difficulty', 'world_number', 'game_status']]
    df_merged = df.merge(df_filtered, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    return df_merged


def get_last_time_steps_of_games(df):
    # get indices of last dataframe row of each game
    last_time_steps = df.groupby(['subject_id', 'game_difficulty', 'world_number']).tail(1)
    return last_time_steps


def get_eyetracker_samples(subject, difficulty, world_number):
    df = read_subject_data(subject)
    df = df[(df['game_difficulty'] == difficulty) & (df['world_number'] == world_number)]
    samples = df[['time', 'gaze_x', 'gaze_y', 'pupil_size']]
    samples = samples.to_numpy()
    return samples
