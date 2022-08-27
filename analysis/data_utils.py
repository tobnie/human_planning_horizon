import pandas as pd


def read_data():
    df = pd.read_csv('./data_compressed_test.gzip', compression='gzip')
    return df


def drop_missing_samples(df):
    df = df.drop(df[df.gaze_x == -32768].index)
    return df


def get_all_gazes(df=None):
    if not df:
        df = read_data()
    return df.loc[['gaze_x', 'gaze_y']]


def get_eyetracker_samples(subject, difficulty, world_number):
    df = read_data()
    df = df[(df['subject_id'] == subject) & (df['game_difficulty'] == difficulty) & (df['world_number'] == world_number)]
    samples = df[['time', 'gaze_x', 'gaze_y', 'pupil_size']]
    samples = samples.to_numpy()
    return samples
