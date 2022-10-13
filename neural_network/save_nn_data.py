import os
import gc

from functools import reduce
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.data_utils import get_all_subjects, create_state_from_string, read_subject_data
from analysis.world.feature_maps import create_feature_map_from_state, get_avoidance_map_from_state_identifier
from neural_network.single_layer_feature_map import create_single_layer_feature_map_from_state


def run_create_IO_data_for_NN(subject_id):
    # get data
    df = pd.read_csv('../data/fixations.csv')
    situations = pd.read_csv('../data/situations.csv')

    situations = situations[situations['subject_id'] == subject_id]
    df = df[df['subject_id'] == subject_id]
    df = df.merge(situations, on=['game_difficulty', 'world_number', 'time', 'player_x_field', 'player_y_field', 'region'],
                  how='left')

    # release memory
    del situations

    df = df[
        ['mfd', 'state', 'fix_x', 'fix_y', 'player_x_field', 'player_y_field', 'fix_x_field', 'fix_y_field', 'state_identifier', 'region']]

    # creating states from df
    print('Creating States from df...')
    states = [create_state_from_string(state_string) for state_string in tqdm(df['state'])]

    print('\nConverting States to Feature Maps...')
    print('Creating single layer fms...')
    # TODO Maybe add target position in single layer fms, such that this information can be used in river lanes near the target?
    state_fms = [create_single_layer_feature_map_from_state(state, None) for state in tqdm(states)]
    print('Creating multi layer fms...')
    state_fms_deep = [create_feature_map_from_state(state) for state in tqdm(states)]

    situations = df['state_identifier'].apply(get_avoidance_map_from_state_identifier)
    situations = np.array(situations.tolist())
    regions = df['region'].apply(lambda x: -1 if x == 'street' else 1 if x == 'river' else 0).to_numpy()

    player_pos = df[['player_x_field', 'player_y_field']].to_numpy()

    # outputs
    gaze_pos = df[['fix_x', 'fix_y']].to_numpy()
    gaze_pos_fields = df[['fix_x_field', 'fix_y_field']].to_numpy()
    mfd = df['mfd'].to_numpy()

    # save all data
    print('Saving data...')
    dir_path = f'../../human_planning_horizon_nn/gaze_predictor/data/{subject_id}/'

    # create directory if non-existent
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savez_compressed(dir_path + 'player_pos.npz', player_pos)
    np.savez_compressed(dir_path + 'single_layer_fm.npz', np.array(state_fms))
    np.savez_compressed(dir_path + 'multi_layer_fm.npz', np.array(state_fms_deep))
    np.savez_compressed(dir_path + 'gaze_pos_continuous.npz', gaze_pos)
    np.savez_compressed(dir_path + 'gaze_pos_discrete.npz', gaze_pos_fields)
    np.savez_compressed(dir_path + 'mfd.npz', mfd)
    np.savez_compressed(dir_path + 'regions.npz', regions)
    np.savez_compressed(dir_path + 'state_3x3.npz', situations[:, 2:-2, 2:-2])
    np.savez_compressed(dir_path + 'state_5x5.npz', situations[:, 1:-1, 1:-1])
    np.savez_compressed(dir_path + 'state_7x7.npz', situations)


def create_IO_data_for_all_subjects():
    # 'normal' data
    for subject in get_all_subjects():
        gc.collect()
        print(f'Creating data for subject {subject}...')
        run_create_IO_data_for_NN(subject)


def create_time_series_data_for_all_subjects(timesteps, stride):
    # time series data for recurrent architectures
    for subject in get_all_subjects():
        df = read_subject_data(subject)
        gc.collect()
        print(f'Creating data for subject {subject}...')
        run_create_IO_data_for_recurrent_NN(df, timesteps, stride)


def run_create_IO_data_for_recurrent_NN(df, timesteps=5, stride=5):
    subject_id = df['subject_id'].values[0]

    # get data
    fix_df = pd.read_csv('../data/fixations.csv')
    fix_df = fix_df[fix_df['subject_id'] == subject_id]
    fix_df = fix_df[['subject_id', 'game_difficulty', 'world_number', 'time', 'mfd', 'state', 'region']]

    situations = pd.read_csv('../data/situations.csv')
    situations = situations[situations['subject_id'] == subject_id]

    merged_df = df.merge(fix_df, on=['subject_id', 'game_difficulty', 'world_number', 'time', 'state', 'region'], how='left')
    merged_df = merged_df.merge(situations, on=['subject_id', 'game_difficulty', 'world_number', 'time', 'region'], how='left')

    # release memory
    del fix_df
    del situations

    df = merged_df[['time', 'mfd', 'state', 'state_identifier', 'region']]

    # position of NaN values in terms of index
    non_null_indices = df.loc[~pd.isna(df['mfd']), :].index
    non_null_indices = list(non_null_indices)

    index_ranges = [list(range(i - stride * (timesteps - 1), i + 1, stride)) if i - stride * (timesteps - 1) > 0 else [] for i in
                    non_null_indices]
    indices = reduce(lambda lst, rnge: lst + rnge, index_ranges)

    indices = np.array(indices)

    five_tail = df.iloc[indices].reset_index(drop=True)
    five_tail_grouped = five_tail.groupby(five_tail.index // timesteps)
    five_tail_groups_filtered = five_tail_grouped.filter(
        lambda sub_frame: sub_frame['time'].is_monotonic_increasing and sub_frame['time'].shape[0] == timesteps)
    df = five_tail_groups_filtered.reset_index(drop=True)
    df_arr = df.to_numpy()

    # creating states from df
    print('Creating States from df...')

    mfd = df_arr[timesteps - 1::timesteps, 1].astype(float)
    states = df_arr[:, 2]
    situations = df_arr[:, 3]

    get_avoidance_map_vectorized = np.vectorize(get_avoidance_map_from_state_identifier)
    situations = np.array([get_avoidance_map_vectorized(situation) for situation in situations])
    regions = df['region'].apply(lambda x: -1 if x == 'street' else 1 if x == 'river' else 0).to_numpy()

    # release memory
    del df_arr
    states = [create_state_from_string(state_string) for state_string in tqdm(states)]

    print('\nConverting States to Feature Maps...')
    print('Creating single layer fms...')

    state_fms = np.array([create_single_layer_feature_map_from_state(state, None) for state in tqdm(states)])

    print('Shapes:\n')
    print('MFD (output):', mfd.shape)
    print('States before Reshaping:', state_fms.shape)

    state_fm_seqs = state_fms.reshape((-1, timesteps, state_fms.shape[-2], state_fms.shape[-1]))
    situation_seqs = situations.reshape((-1, timesteps, situations.shape[-2], situations.shape[-1]))
    region_seqs = regions.reshape((-1, timesteps))

    print('Creating multi layer fms...')
    state_fm_deep_seqs = np.array([create_feature_map_from_state(state) for state in tqdm(states)])
    state_fm_deep_seqs = state_fm_deep_seqs.reshape(
        (-1, timesteps, state_fm_deep_seqs.shape[-3], state_fm_deep_seqs.shape[-2], state_fm_deep_seqs.shape[-1]))

    # # save all data
    print('Saving data...')
    dir_path = f'../../human_planning_horizon_nn/gaze_predictor/data/{subject_id}/lstm/timesteps={timesteps}_stride={stride}/'

    # create directory if non-existent
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savez_compressed(dir_path + 'single_layer_fm_seq.npz', state_fm_seqs)
    np.savez_compressed(dir_path + 'multi_layer_fm_seq.npz'.format(timesteps, stride), state_fm_deep_seqs)
    np.savez_compressed(dir_path + 'mfd_seq.npz'.format(timesteps, stride), mfd)
    np.savez_compressed(dir_path + 'situation_seq_7x7.npz'.format(timesteps, stride), situation_seqs)
    np.savez_compressed(dir_path + 'situation_seq_5x5.npz'.format(timesteps, stride), situation_seqs[:, :, 1:-1, 1:-1])
    np.savez_compressed(dir_path + 'situation_seq_3x3.npz'.format(timesteps, stride), situation_seqs[:, :, 2:-2, 2:-2])
    np.savez_compressed(dir_path + 'region_seq.npz'.format(timesteps, stride), region_seqs)


if __name__ == '__main__':

    print('NON-RECURRENT DATA')
    # data for 'normal' architectures
    create_IO_data_for_all_subjects()

    print('RECURRENT DATA')
    # data for recurrent architectures
    stride = reversed([20, 10, 5])
    prev_timesteps = [5, 20, 50, 100]

    for s, t_steps in tqdm(product(stride, prev_timesteps)):
        # create normal NN data
        print(f'Running with stride={s}, time_steps={t_steps}')
        create_time_series_data_for_all_subjects(timesteps=t_steps, stride=s)
