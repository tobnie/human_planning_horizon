import csv
import os

import numpy as np

import config


def read_csv(path):
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        csv_dict = {rows[0]: rows[1] for rows in reader}
        return csv_dict


def get_level_data_path(subject, difficulty, world_name, training=False):
    return config.LEVEL_DATA_DIR + f'{subject}/' + ('training/' if training else '') + f'{difficulty}/{world_name}/'


def get_world_properties_path(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'world_properties.csv'


def get_world_properties(subject, difficulty, world_name, training=False):
    path = get_world_properties_path(subject, difficulty, world_name, training)  # TODO remove absolute path
    path = 'D:/source/human_planning_horizon/data/level_data/TEST01/training/easy/world_0/world_properties.csv'
    return read_csv(path)


def get_state_data_path(subject, difficulty, world_name):
    return get_level_data_path(subject, difficulty, world_name) + 'states/'


def get_time_from_state_file(state_file):
    return int(state_file.split('.')[0].split('_')[1])


def get_times_states(subject, difficulty, world_name):
    """ Loads all world states as np array representation for given subject, difficulty and world_name."""
    # get all state array
    # TODO why only working with absolute path?
    states_path = get_state_data_path(subject, difficulty, world_name)
    states_path = 'D:/source/human_planning_horizon/data/level_data/TEST01/training/easy/world_0/states/'
    state_files = [f for f in os.listdir(states_path) if f.endswith(".npz")]

    # load array for each
    times_states = []
    for f in state_files:
        time = get_time_from_state_file(f)
        state = np.load(states_path + f)['arr_0']
        times_states.append((time, state))

    # make times to ints
    times_states = [(int(time), state) for time, state in times_states]

    # sort by time
    times_states = sorted(times_states, key=lambda x: x[0])

    return times_states


def assign_position_to_fields(x, y, width):
    field_y = int(round(y / config.FIELD_HEIGHT))
    field_x_start = int(round(x / config.FIELD_WIDTH))
    field_width = int(width // config.FIELD_WIDTH)
    return field_x_start, field_y, field_width


def create_feature_map_from_state(state, target_position):
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES, 4))
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

        feature_map[x_start:x_start + width, y, obj_type] = 1

    # add target position
    feature_map[target_position, config.N_LANES - 1, 3] = 1
    feature_map = np.rot90(feature_map)
    return feature_map

