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


if __name__ == '__main__':
    run_create_IO_data_for_NN()
