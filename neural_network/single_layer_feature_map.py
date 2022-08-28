import numpy as np
from tqdm import tqdm

import config
from analysis.data_utils import read_data, read_subject_data, create_state_from_string
from data.save_data_compressed import create_feature_map_from_state, assign_position_to_fields


def create_state_from_str(string):
    return np.fromstring(string)


# player: 1, vehicle: 2, lilypad: 3
def create_single_layer_feature_map_from_state(state):
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES))

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    for obj_type, x, y, width in state:
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


# def create_single_layer_feature_map_from_state(state):
#     # create 'deep' feature map
#     normal_fm = create_feature_map_from_state(state)
#
#     single_layer_fm = np.zeros_like(normal_fm[:, :, 0])
#
#     # transform into feature_map where number encodes object instead of depth
#     for depth_i in reversed(range(normal_fm.shape[-1])):
#         obj_positions = np.where(normal_fm[:, :, depth_i] == 1)
#
#         # by the following we can recognize collisions in a feature map by a value > 10
#         # TODO do we want it that way? we could keep it. if we discard it, the player value is written last, such that the underlying collision is hidden
#         single_layer_fm[obj_positions] = single_layer_fm[obj_positions] * 10 + depth_i + 1
#
#     return single_layer_fm


def get_inputs_outputs_for_nn(df):
    df = df[['gaze_x', 'gaze_y', 'state']]

    # creating states from df
    print('Creating States from df...')
    states = [create_state_from_string(state_string) for state_string in tqdm(df['state'])]

    print('\nConverting States to Feature Maps...')
    state_fms = [create_single_layer_feature_map_from_state(state) for state in tqdm(states)]
    gaze_pos = df[['gaze_x', 'gaze_y']].to_numpy()

    inputs = state_fms
    outputs = gaze_pos
    return inputs, outputs


def save_inputs_output_for_training_of_nn(inputs, outputs):
    print("\nSaving data in input and output file...")
    # flatten input feature maps
    flattened_states = [np.ravel(state) for state in inputs]

    # save inputs and outputs as .npz
    np.savez_compressed('../neural_network/input.npz', flattened_states)
    np.savez_compressed('../neural_network/output.npz', outputs)

    print("Done!")


def run_create_IO_data_for_NN():
    # TODO run for all subjects
    df = read_data()
    inputs, outputs = get_inputs_outputs_for_nn(df)
    save_inputs_output_for_training_of_nn(inputs, outputs)

    # example_state = np.array(df['state'][0])
    # fm_test = create_single_layer_feature_map_from_state(example_state)
    # print(fm_test)
