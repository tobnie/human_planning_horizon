import numpy as np
from tqdm import tqdm

import config
from analysis.data_utils import read_data, read_subject_data, create_state_from_string, get_only_onscreen_data
from data.save_data_compressed import assign_position_to_fields, create_feature_map_from_state

TARGET_POS2DISCRETE = {448: -1, 1216: 0, 2112: 1}


def create_state_from_str(string):
    return np.fromstring(string)


def create_single_layer_feature_map_from_state(state, target_pos=None):
    """ Creates a feature map consisting of layer with the following encodings:
    player: 1, vehicle: 2, lilypad: 3, target_position: 4
    """
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES))

    # encode target position
    if target_pos is not None:
        x_target_pos = int(round(target_pos / config.FIELD_WIDTH))
        y_target_pos = config.N_LANES - 1
        feature_map[x_target_pos, y_target_pos] = 4

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    for obj_type, x, y, width in reversed(state):
        x_start, y, width = assign_position_to_fields(x, y, width)

        # correct player width
        if obj_type == 0:
            width = 1

        # correct for partially visible obstacles
        if x_start < 0:
            width = width + x_start
            x_start = 0

        if x_start + width > config.N_FIELDS_PER_LANE:
            width = config.N_FIELDS_PER_LANE - x_start

        feature_map[x_start:x_start + width, y] = obj_type + 1

    feature_map = np.rot90(feature_map)

    # invert y axis
    feature_map = np.flip(feature_map, axis=1)

    return feature_map


def get_inputs_outputs_for_nn_including_target_position(df, single_layer_fm):
    df = df[['gaze_x', 'gaze_y', 'state', 'target_position']]

    df = get_only_onscreen_data(df)

    target_pos = df['target_position'].apply(lambda x: TARGET_POS2DISCRETE[x]).to_numpy()

    # creating states from df
    print('Creating States from df...')
    states = [create_state_from_string(state_string) for state_string in tqdm(df['state'])]

    print('\nConverting States to Feature Maps...')
    if single_layer_fm:
        state_fms = [create_single_layer_feature_map_from_state(state, p_target) for state, p_target in tqdm(zip(states, target_pos))]
    else:
        raise NotImplementedError("Not yet implemented")
        # state_fms = [create_feature_map_from_state(state) for state in tqdm(states)]
    gaze_pos = df[['gaze_x', 'gaze_y']].to_numpy()

    inputs = np.array(state_fms)
    outputs = gaze_pos
    return inputs, outputs, target_pos


def get_inputs_outputs_for_nn(df, single_layer_fm):
    df = df[['gaze_x', 'gaze_y', 'state', 'target_position']]

    df = get_only_onscreen_data(df)

    # creating states from df
    print('Creating States from df...')
    states = [create_state_from_string(state_string) for state_string in tqdm(df['state'])]

    print('\nConverting States to Feature Maps...')
    if single_layer_fm:
        state_fms = [create_single_layer_feature_map_from_state(state) for state in tqdm(states)]
    else:
        state_fms = [create_feature_map_from_state(state) for state in tqdm(states)]
    gaze_pos = df[['gaze_x', 'gaze_y']].to_numpy()

    target_pos = df['target_position'].apply(lambda x: TARGET_POS2DISCRETE[x]).to_numpy()

    inputs = np.array(state_fms)
    outputs = gaze_pos
    return inputs, outputs, target_pos


def save_inputs_output_for_training_of_nn(inputs, outputs, target_pos=None, suffix=''):
    print("\nSaving data in input and output file...")

    # save inputs and outputs as .npz
    if suffix != '':
        suffix = '_' + suffix

    file_name_in = f'input{suffix}.npz'
    np.savez_compressed(f'../neural_network/{file_name_in}', inputs)
    np.savez_compressed(f'../neural_network/output.npz', outputs)

    if target_pos is not None:
        np.savez_compressed(f'../neural_network/target_pos.npz', target_pos)

    print("Done!")


def run_create_IO_data_for_NN():
    df = read_data()

    inputs, outputs, target_pos = get_inputs_outputs_for_nn(df, single_layer_fm=True)
    save_inputs_output_for_training_of_nn(inputs, outputs, target_pos=target_pos)

    inputs, outputs, target_pos = get_inputs_outputs_for_nn(df, single_layer_fm=False)
    save_inputs_output_for_training_of_nn(inputs, outputs, target_pos=target_pos, suffix='deep_fm')

    inputs, outputs, target_pos = get_inputs_outputs_for_nn_including_target_position(df, single_layer_fm=True)
    save_inputs_output_for_training_of_nn(inputs, outputs, suffix='including_target_pos')

    # TODO
    # inputs, outputs, target_pos = get_inputs_outputs_for_nn_including_target_position(df, single_layer_fm=False)
    # save_inputs_output_for_training_of_nn(inputs, outputs, suffix='including_target_pos_deep')