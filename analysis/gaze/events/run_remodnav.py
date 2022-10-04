import numpy as np
import pandas as pd
from remodnav import EyegazeClassifier

import config
from analysis.data_utils import coords2fieldsx, coords2fieldsy, read_data
from analysis.gaze.adaptiveIDT.data_filtering import PIX2DEG
from analysis.gaze.fixations import attribute_fixations_in_df


def run_remodnav_as_method(data):
    # TODO Choose filter parameters such that there are no warnings anymore?
    # parameters
    sampling_rate = 60
    MIN_SACCADE_DURATION = 20 * 1e-3
    savgol_length = 3 * MIN_SACCADE_DURATION

    # remodnav in action
    clf = EyegazeClassifier(px2deg=PIX2DEG, sampling_rate=sampling_rate, min_saccade_duration=MIN_SACCADE_DURATION)
    pp = clf.preproc(data, savgol_length=savgol_length)
    events = clf(pp, classify_isp=True, sort_events=True)

    return events


def run_remodnav(trial_df):
    # format for remodnav algorithm
    gaze_df = trial_df.copy()[['gaze_x', 'gaze_y']]
    gaze_df.columns = ['x', 'y']
    result_as_list_of_dicts = run_remodnav_as_method(gaze_df.to_records())
    result_df = pd.DataFrame(result_as_list_of_dicts).drop('id', axis=1)

    # add duration
    result_df['duration'] = result_df['end_time'] - result_df['start_time']

    # add level information
    result_df = attribute_player_positions_to_time(trial_df, result_df)

    return result_df


def reset_time_of_df(df):
    min_time = df['time'].min()
    df['time'] = df['time'] - min_time
    return df


def add_values_for_closest_time(time, trial_df, remodnav_results):
    trial_df = reset_time_of_df(trial_df)
    idx_closest_time = find_index_for_closest_value(trial_df['time'], time * 1_000)  # convert to ms

    time_mask_remodnav = remodnav_results['start_time'] == time
    remodnav_row = remodnav_results[time_mask_remodnav]

    time = trial_df.loc[idx_closest_time]['time']
    player_x = trial_df.loc[idx_closest_time]['player_x']
    player_y = trial_df.loc[idx_closest_time]['player_y']
    player_x_field = trial_df.loc[idx_closest_time]['player_x_field']
    player_y_field = trial_df.loc[idx_closest_time]['player_y_field']
    state = trial_df.loc[idx_closest_time]['state']
    score = trial_df.loc[idx_closest_time]['score']
    region = trial_df.loc[idx_closest_time]['region']
    target_position = trial_df.loc[idx_closest_time]['target_position']

    remodnav_new = {
        'time': time,
        'player_x': player_x,
        'player_y': player_y,
        'player_x_field': player_x_field,
        'player_y_field': player_y_field,
        'label': remodnav_row['label'].values[0],
        'duration': remodnav_row['duration'].values[0],
        'start_x': remodnav_row['start_x'].values[0],
        'start_y': remodnav_row['start_y'].values[0],
        'end_x': remodnav_row['end_x'].values[0],
        'end_y': remodnav_row['end_y'].values[0],
        'region': region,
        'score': score,
        'state': state,
        'target_position': target_position
    }

    return remodnav_new


def attribute_player_positions_to_time(trial_df, remodnav_results):
    test = remodnav_results.apply(lambda x: add_values_for_closest_time(x['start_time'], trial_df, remodnav_results), axis=1)
    test = test.to_list()
    test = pd.DataFrame(test)
    return test


def find_index_for_closest_value(series, value):
    idx = (series - value).abs().idxmin()
    return idx


def add_mfd_to_remodnav(df):
    total_fix_duration = df['duration'].sum()
    df['mfd'] = (df['fix_distance_manhattan'] * df['duration']).sum() / total_fix_duration
    return df


def add_mfa_to_remodnav(df):
    total_fix_duration = df['duration'].sum()
    df['mfa'] = (df['fix_angle'] * df['duration']).sum() / total_fix_duration
    return df


def save_remodnav_fixations(drop_offscreen_samples=True):
    missing_threshold = -30000
    df = read_data()
    results = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(run_remodnav).reset_index().drop(columns='level_3')
    print(results)

    fixations = results[results['label'] == 'FIXA']

    # throw off invalid fixations
    fixations = fixations[
        (fixations.end_x > missing_threshold) & (fixations.end_y > missing_threshold) & (fixations.start_x > missing_threshold) & (
                fixations.start_y > missing_threshold)]

    # middle between start and end as fixation coordinate
    fixations['fix_x'] = fixations.apply(lambda x: (x['start_x'] + x['end_x']) / 2, axis=1)
    fixations['fix_y'] = fixations.apply(lambda x: (x['start_y'] + x['end_y']) / 2, axis=1)

    # into field coordinates
    fixations['fix_x_field'] = fixations['fix_x'].apply(coords2fieldsx)
    fixations['fix_y_field'] = fixations['fix_y'].apply(coords2fieldsy)
    fixations['fix_distance_manhattan'] = fixations.apply(
        lambda x: abs(x['fix_x_field'] - x['player_x_field']) + abs(x['fix_y_field'] - x['player_y_field']), axis=1)
    fixations['fix_angle'] = fixations.apply(
        lambda x: np.arctan2(x['fix_y_field'] - x['player_y_field'], x['fix_x_field'] - x['player_x_field']), axis=1)

    fixations = fixations.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field']).apply(
        add_mfd_to_remodnav)
    fixations = fixations.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field']).apply(
        add_mfa_to_remodnav)
    fixations = fixations.drop_duplicates()

    fixations.rename(columns={'duration': 'fix_duration'}, inplace=True)

    if drop_offscreen_samples:
        mask = (fixations['fix_x'] <= config.DISPLAY_WIDTH_PX) & (fixations['fix_x'] >= 0) & (fixations['fix_y'] >= 0) & (
                    fixations['fix_y'] <= config.DISPLAY_HEIGHT_PX)
        fixations = fixations[mask]

    fixations = attribute_fixations_in_df(fixations)
    fixations.to_csv('../data/fixations_remodnav.csv', index=False)


if __name__ == '__main__':
    save_remodnav_fixations()
