import numpy as np
from tqdm import tqdm

from analysis.data_utils import read_data, get_only_onscreen_data, create_state_from_string
from data.preprocessing import create_feature_map_from_state
from neural_network.single_layer_feature_map import TARGET_POS2DISCRETE, create_single_layer_feature_map_from_state


def run_create_IO_data_for_NN():
    # get data
    df = read_data()
    df = get_only_onscreen_data(df)
    df = df[['gaze_x', 'gaze_y', 'state', 'target_position', 'gaze_x_field', 'gaze_y_field']]

    # creating states from df
    print('Creating States from df...')
    states = [create_state_from_string(state_string) for state_string in tqdm(df['state'])]

    print('\nConverting States to Feature Maps...')
    print('Creating single layer fms...')
    # TODO state_fms do not contain target_pos for now!
    state_fms = [create_single_layer_feature_map_from_state(state, None) for state in tqdm(states)]
    print('Creating multi layer fms...')
    state_fms_deep = [create_feature_map_from_state(state) for state in tqdm(states)]

    target_pos = df['target_position'].apply(lambda x: TARGET_POS2DISCRETE[x]).to_numpy()

    # outputs
    gaze_pos = df[['gaze_x', 'gaze_y']].to_numpy()
    gaze_pos_fields = df[['gaze_x_field', 'gaze_y_field']].to_numpy()

    # save all data
    print('Saving data...')
    # D:\source\human_planning_horizon_nn\gaze_predictor\gaze_predictor\data
    dir_path = '../../human_planning_horizon_nn/gaze_predictor/gaze_predictor/data/'
    np.savez_compressed(dir_path + 'single_layer_fm.npz', np.array(state_fms))
    np.savez_compressed(dir_path + 'multi_layer_fm.npz', np.array(state_fms_deep))
    np.savez_compressed(dir_path + 'output_continuous.npz', gaze_pos)
    np.savez_compressed(dir_path + 'output_discrete.npz', gaze_pos_fields)
    np.savez_compressed(dir_path + 'target_pos.npz', target_pos)


if __name__ == '__main__':
    run_create_IO_data_for_NN()
