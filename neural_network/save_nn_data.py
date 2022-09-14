from functools import reduce

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.data_utils import read_data, get_only_onscreen_data, create_state_from_string
from data.preprocessing import create_feature_map_from_state
from neural_network.single_layer_feature_map import TARGET_POS2DISCRETE, create_single_layer_feature_map_from_state


def run_create_IO_data_for_NN():
    # get data
    df = pd.read_csv('../data/fixations.csv')
    df = df[['weighted_fix_distance_euclidean',
             'weighted_fix_distance_manhattan', 'state', 'fix_x', 'fix_y', 'fix_x_field', 'fix_y_field', 'fix_distance_manhattan',
             'fix_distance_euclidean']]

    # creating states from df
    print('Creating States from df...')
    states = [create_state_from_string(state_string) for state_string in tqdm(df['state'])]

    print('\nConverting States to Feature Maps...')
    print('Creating single layer fms...')
    # TODO state_fms do not contain target_pos for now!
    state_fms = [create_single_layer_feature_map_from_state(state, None) for state in tqdm(states)]
    print('Creating multi layer fms...')
    state_fms_deep = [create_feature_map_from_state(state) for state in tqdm(states)]

    # outputs
    gaze_pos = df[['fix_x', 'fix_y']].to_numpy()
    gaze_pos_fields = df[['fix_x_field', 'fix_y_field']].to_numpy()
    weighted_distance_manhattan = df['weighted_fix_distance_manhattan'].to_numpy()
    weighted_distance_euclidean = df['weighted_fix_distance_euclidean'].to_numpy()

    # save all data
    print('Saving data...')
    dir_path = '../../human_planning_horizon_nn/gaze_predictor/gaze_predictor/data/'
    np.savez_compressed(dir_path + 'single_layer_fm.npz', np.array(state_fms))
    np.savez_compressed(dir_path + 'multi_layer_fm.npz', np.array(state_fms_deep))

    np.savez_compressed(dir_path + 'gaze_pos_continuous.npz', gaze_pos)
    np.savez_compressed(dir_path + 'gaze_pos_discrete.npz', gaze_pos_fields)
    np.savez_compressed(dir_path + 'mfd.npz', weighted_distance_manhattan)
    np.savez_compressed(dir_path + 'mfd_euclid.npz', weighted_distance_euclidean)


def run_create_IO_data_for_recurrent_NN(prev_timesteps=5):
    # get data
    df = read_data()
    fix_df = pd.read_csv('../data/fixations.csv')
    fix_df = fix_df[['subject_id', 'game_difficulty', 'world_number', 'time', 'weighted_fix_distance_manhattan', 'state']]
    merged_df = df.merge(fix_df, on=['subject_id', 'game_difficulty', 'world_number', 'time', 'state'], how='left')
    df = merged_df[['time', 'weighted_fix_distance_manhattan', 'state']]

    # position of NaN values in terms of index
    non_null_indices = df.loc[~pd.isna(df['weighted_fix_distance_manhattan']), :].index
    # non_null_indices = (~df['weighted_fix_distance_manhattan'].isna()).index
    non_null_indices = list(non_null_indices)

    index_ranges = [list(range(i - prev_timesteps + 1, i + 1)) for i in non_null_indices]
    indices = reduce(lambda lst, rnge: lst + rnge, index_ranges)

    indices = np.array(indices)
    indices = indices[indices >= prev_timesteps]

    five_tail = df.iloc[indices].reset_index(drop=True)
    five_tail_grouped = five_tail.groupby(five_tail.index // 5)
    five_tail_groups_filtered = five_tail_grouped.filter(lambda sub_frame: sub_frame['time'].is_monotonic)
    df = five_tail_groups_filtered.reset_index(drop=True)
    df_arr = df.to_numpy()

    # creating states from df
    print('Creating States from df...')
    test = [create_state_from_string(state_string) for state_string in tqdm(df_arr[:, :, -1])]
    df_arr[:, :, -1] = test
    # TODO
    df_arr = df_arr.reshape((prev_timesteps, -1, 3))

    print('\nConverting States to Feature Maps...')
    print('Creating single layer fms...')

    # # TODO put into sequences
    # state_fm_seqs = [create_single_layer_feature_map_from_state(state, None) for state in tqdm(states)]
    # print('Creating multi layer fms...')
    # state_fm_deep_seqs = [create_feature_map_from_state(state) for state in tqdm(states)]

    # # save all data
    # print('Saving data...')
    # dir_path = '../../human_planning_horizon_nn/gaze_predictor/gaze_predictor/data/'
    # np.savez_compressed(dir_path + 'single_layer_fm_seq.npz', np.array(state_fm_seqs))
    # np.savez_compressed(dir_path + 'multi_layer_fm_seq.npz', np.array(state_fm_deep_seqs))


if __name__ == '__main__':
    run_create_IO_data_for_recurrent_NN()
    # run_create_IO_data_for_NN()
