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


def get_states(subject, difficulty, world_name):
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

    return times_states
