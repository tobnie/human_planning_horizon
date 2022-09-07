import os
import string

import numpy as np
import pandas as pd

import config
from analysis.score.recalculate_score import add_estimated_scores_when_missing

WIN_THRESHOLD_Y = (config.N_LANES - 1) * config.FIELD_HEIGHT
TIME_OUT_THRESHOLD = config.LEVEL_TIME - config.PLAYER_UPDATE_INTERVAL

SKIP_SUBJECTS = ['MA02CA', 'PE10MI', 'ZI01SU', 'NI07LU']


def get_all_subjects():
    data_dir = f'../data/level_data/'
    subjects = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir + f) if f not in SKIP_SUBJECTS]
    return subjects


def read_data():
    subject_dfs = []
    for subject in get_all_subjects():
        subject_df = read_subject_data(subject)
        subject_dfs.append(subject_df)
    df = pd.concat(subject_dfs)
    df = add_estimated_scores_when_missing(df)
    return df


def create_state_from_string(instr):
    instr = instr.replace("[", "").replace("]", "")  # .split(",")
    return np.fromstring(instr, sep=', ', dtype=int).reshape((-1, 4))


def read_subject_data(subject_id):
    df = pd.read_csv(f'../data/compressed_data/{subject_id}_compressed.gzip',
                     compression='gzip')  # , converters={'state': create_state_from_string})
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


def get_only_onscreen_data(df):
    """ Returns the df, but only rows where the gaze was on screen"""
    gaze_on_screen_mask = (0.0 <= df['gaze_x']) & (df['gaze_x'] <= config.DISPLAY_WIDTH_PX) & (0.0 <= df['gaze_y']) & (
            df['gaze_y'] <= config.DISPLAY_HEIGHT_PX)
    return df[gaze_on_screen_mask]


def drop_missing_samples(df):
    df = df.drop(df[df.gaze_x == -32768].index)
    return df


def get_all_gazes(df=None):
    if not df:
        df = read_data()
    return df.loc[['gaze_x', 'gaze_y']]


def return_status(time, player_y, player_x, target_position):
    # player in last sample was in second to last row and player center was below target field
    x_win_range = (target_position - config.FIELD_WIDTH / 2, target_position + config.FIELD_WIDTH / 2)
    if x_win_range[0] <= player_x <= x_win_range[1] and player_y >= 13 * config.FIELD_HEIGHT:
        if time < TIME_OUT_THRESHOLD:
            return 'won'
        else:
            return 'timed_out'
    else:
        if time < TIME_OUT_THRESHOLD:
            return 'lost'
        else:
            return 'timed_out'


def add_game_status_to_df(df):
    last_time_steps = get_last_time_steps_of_games(df).copy()

    last_time_steps['game_status'] = last_time_steps.apply(
        lambda x: return_status(x['time'], x['player_y'], x['player_x'], x['target_position']), axis=1)
    df_filtered = last_time_steps[['subject_id', 'game_difficulty', 'world_number', 'game_status']]
    df_merged = df.merge(df_filtered, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    return df_merged


def get_street_data(df):
    street_mask = (df['player_y_field'] >= 1) & (df['player_y_field'] <= 6)
    df_street = df[street_mask]
    return df_street


def get_river_data(df):
    river_mask = (df['player_y_field'] >= 8) & (df['player_y_field'] <= 13)
    df_river = df[river_mask]
    return df_river


def get_last_time_steps_of_games(df=None):
    if df is None:
        df = read_data()

    # get indices of last dataframe row of each game
    last_time_steps = df.groupby(['subject_id', 'game_difficulty', 'world_number']).tail(1)
    return last_time_steps


def get_eyetracker_samples(subject, difficulty, world_number):
    df = read_subject_data(subject)
    df = df[(df['game_difficulty'] == difficulty) & (df['world_number'] == world_number)]
    samples = df[['time', 'gaze_x', 'gaze_y', 'pupil_size']]
    samples = samples.to_numpy()
    return samples


def transform_target_pos_to_string(df):
    pos2str = {448: 'left', 1216: 'center', 2112: 'right'}
    df['target_position'] = df['target_position'].apply(lambda x: pos2str[x]).copy()
    return df


def position2field(x, y):
    field_x = int(round(x / config.FIELD_WIDTH))
    field_y = int(round(y / config.FIELD_HEIGHT))
    return field_x, field_y


def assign_position_to_fields(x, y, width):
    field_x_start, field_y = position2field(x, y)
    field_width = int(width // config.FIELD_WIDTH)
    return field_x_start, field_y, field_width


def field2screen(field_coords):
    FIELD_WIDTH = config.FIELD_WIDTH
    FIELD_HEIGHT = config.FIELD_HEIGHT
    screen_coords = [(x * FIELD_WIDTH + FIELD_WIDTH / 2, y * FIELD_HEIGHT + FIELD_HEIGHT / 2) for x, y in field_coords]
    return np.array(screen_coords)


def subject2letter(subject_id):
    unique_subjects = get_all_subjects()
    for i, subj_id in enumerate(unique_subjects):
        if subject_id == subj_id:
            return string.ascii_uppercase[i]
    return None
