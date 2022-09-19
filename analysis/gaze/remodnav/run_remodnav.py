import pandas as pd
from remodnav import EyegazeClassifier

from analysis.data_utils import read_subject_data, read_data
from analysis.gaze.adaptiveIDT.data_filtering import PIX2DEG

from analysis.player.player_position_heatmap import coords2fieldsx, coords2fieldsy


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

    player_x_field = trial_df.loc[idx_closest_time]['player_x_field']
    player_y_field = trial_df.loc[idx_closest_time]['player_y_field']

    remodnav_new = {
        'time': time,
        'player_x_field': player_x_field,
        'player_y_field': player_y_field,
        'label': remodnav_row['label'].values[0],
        'duration': remodnav_row['duration'].values[0],
        'start_x': remodnav_row['start_x'].values[0],
        'start_y': remodnav_row['start_y'].values[0],
        'end_x': remodnav_row['end_x'].values[0],
        'end_y': remodnav_row['end_y'].values[0],
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
    df['mfd'] = (df['manhattan_distance'] * df['duration']).sum() / total_fix_duration
    return df


def get_fixations_from_remodnav():
    missing_threshold = -30000
    df = read_data()
    results = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(run_remodnav).reset_index().drop(columns='level_3')
    print(results)

    fixations = results[results['label'] == 'FIXA']

    # throw off invalid fixations
    fixations = fixations[
        (fixations.end_x > missing_threshold) & (fixations.end_y > missing_threshold) & (fixations.start_x > missing_threshold) & (
                fixations.start_y > missing_threshold)]
    fixations['center_x'] = fixations.apply(lambda x: (x['start_x'] + x['end_x']) / 2, axis=1)
    fixations['center_y'] = fixations.apply(lambda x: (x['start_y'] + x['end_y']) / 2, axis=1)

    # into field coordinates
    fixations['center_x_field'] = fixations['center_x'].apply(coords2fieldsx)
    fixations['center_y_field'] = fixations['center_y'].apply(coords2fieldsy)
    fixations['manhattan_distance'] = fixations.apply(
        lambda x: abs(x['center_x_field'] - x['player_x_field']) + abs(x['center_y_field'] - x['player_y_field']), axis=1)

    fixations = fixations.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field']).apply(add_mfd_to_remodnav)
    fixations = fixations.drop_duplicates()

    print('New Method:')
    print(fixations)

    fixations.to_csv('../data/fixations_remodnav.csv', index=False)
    #
    # old_fixations = pd.read_csv('../data/fixations.csv')
    # old_fixations = old_fixations[(old_fixations['subject_id'] == subject_id) & (old_fixations['game_difficulty'] == game_difficulty) & (
    #             old_fixations['world_number'] == world_number)]
    # old_fixations = old_fixations[['player_x_field', 'player_y_field', 'weighted_fix_distance_manhattan']]
    # old_fixations = old_fixations.drop_duplicates()
    # print('Old Method:')
    # print(old_fixations)


if __name__ == '__main__':
    get_fixations_from_remodnav()

