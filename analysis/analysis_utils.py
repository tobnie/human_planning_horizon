import csv
import os

import numpy as np
import pandas as pd

import config
from analysis.plotting.world.world_coordinates import get_center_of_entity
from game.world_generation.generation_config import GameDifficulty

OBJECT_TO_INT = {
    'player': 0,
    'vehicle': 1,
    'lilypad': 2
}

ACTION_TO_STRING = {
    0: 'right',
    1: 'left',
    2: 'up',
    3: 'down',
    4: 'nop'
}


def read_csv(path):
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        csv_dict = {rows[0]: rows[1] for rows in reader}
        return csv_dict


def read_npz(path):
    try:
        return np.load(path)['arr_0']
    except FileNotFoundError:
        return []


def get_level_data_path(subject, difficulty, world_name, training=False):
    return f'../data/level_data/{subject}/' + ('training/' if training else '') + f'{difficulty}/{world_name}/'


def get_samples_data_path(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'eyetracker_samples.npz'


def do_something_for_all_worlds(subject_id, func):
    result_acc = []
    for difficulty in GameDifficulty:
        for i in range(20):
            single_result = func(subject_id, difficulty.value, f'world_{i}')
            result_acc.extend(single_result)
    return result_acc


def get_all_samples_for_subject(subject):
    return do_something_for_all_worlds(subject, get_eyetracker_samples)


def get_eyetracker_events_data_path(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'eyetracker_events.npz'


def get_world_properties_path(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'world_properties.csv'


def get_world_properties(subject, difficulty, world_name, training=False):
    path = get_world_properties_path(subject, difficulty, world_name, training)
    return read_csv(path)


def get_state_data_path(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'states/'


def get_actions_path(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'actions.npz'


def get_time_from_state_file(state_file):
    return int(state_file.split('.')[0].split('_')[1])


def get_eyetracker_samples(subject, difficulty, world_name, training=False):
    path = get_samples_data_path(subject, difficulty, world_name, training)
    return read_npz(path)


def get_eyetracker_events(subject, difficulty, world_name, training=False):
    path = get_eyetracker_events_data_path(subject, difficulty, world_name, training)
    return read_npz(path)


def get_state_path_single_npz(subject, difficulty, world_name, training=False):
    return get_level_data_path(subject, difficulty, world_name, training) + 'world_states.npz'


def get_times_states_from_single_npz(subject, difficulty, world_name, training=False):
    npz_file = np.load(get_state_path_single_npz(subject, difficulty, world_name, training), allow_pickle=True)
    times = npz_file['times']
    states = npz_file['states']
    times_states = list(zip(times, states))
    return times_states


def get_states(subject, difficulty, world_name, training=False):
    times_states = get_times_states(subject, difficulty, world_name, training)
    times, states = list(zip(*times_states))
    return states


def get_times_states_from_folder(subject, difficulty, world_name, training=False):
    try:
        # get all state array
        states_path = get_state_data_path(subject, difficulty, world_name, training=training)
        state_files = [f for f in os.listdir(states_path) if f.endswith(".npz")]

        # load array for each
        times_states = []
        for f in state_files:
            time = get_time_from_state_file(f)
            state = np.load(states_path + f)['arr_0']
            times_states.append((time, state))
    except FileNotFoundError:
        return []

    return times_states


def get_times_states(subject, difficulty, world_name, training=False):
    """ Loads all world states as np array representation for given subject, difficulty and world_name."""

    try:
        times_states = get_times_states_from_single_npz(subject, difficulty, world_name, training=training)
    except FileNotFoundError:
        times_states = get_times_states_from_folder(subject, difficulty, world_name, training=training)

    # make times to ints
    times_states = [(int(time), state) for time, state in times_states]

    # sort by time
    times_states = sorted(times_states, key=lambda x: x[0])

    return times_states


def get_times_actions(subject, difficulty, world_name, training=False):
    """ Loads all world states as np array representation for given subject, difficulty and world_name."""

    # get all state array
    actions_path = get_actions_path(subject, difficulty, world_name, training=training)

    try:
        times_actions = np.load(actions_path)['arr_0']
    except FileNotFoundError:
        return []

    # make times to ints
    times_actions = [(int(time), int(action)) for time, action in times_actions]

    # sort by time
    times_actions = sorted(times_actions, key=lambda x: x[0])

    return times_actions


def get_all_times_states_of_player(subject_id):
    return do_something_for_all_worlds(subject_id, get_times_states)


def get_all_times_actions_of_player(subject_id):
    return do_something_for_all_worlds(subject_id, get_times_actions)


def get_times_actions_states(subject_id):
    times_actions = get_all_times_actions_of_player(subject_id)
    times_states = get_all_times_states_of_player(subject_id)

    t_actions = list(zip(*times_actions))[0]

    times_actions_states = [(time, action, state) if time in t_actions else () for (time, action), (_, state) in
                            zip(times_actions, times_states)]
    return times_actions_states


def filter_times_actions_states_by_action(times_action_states):
    """ Filters a given collection of time-action-state-tuples by action.
    Returns a dict containing all states in which the respective action was taken."""
    states_dict = {
        'up': [],
        'down': [],
        'left': [],
        'right': [],
        'nop': []
    }
    for t, a, s in times_action_states:
        states_dict[ACTION_TO_STRING[a]].append(s)

    return states_dict


def get_all_subjects():
    data_dir = f'../data/level_data/'
    subjects = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir + f)]
    return subjects


def get_all_levels_for_subject(subject):
    all_difficulties = []
    world = []
    all_states = []
    all_actions = []
    all_times_s = []
    all_times_a = []
    for difficulty in GameDifficulty:
        for i in range(20):
            times_states = get_times_states(subject, difficulty.value, f'world_{i}')
            times_actions = get_times_actions(subject, difficulty.value, f'world_{i}')

            if not times_states:
                continue

            times_s, states = list(zip(*times_states))
            times_a, actions = list(zip(*times_actions))

            all_difficulties.append(difficulty.value)
            world.append(i)
            all_times_s.append(times_s)
            all_times_a.append(times_a)
            all_states.append(states)
            all_actions.append(actions)
    return all_difficulties, world, all_times_s, all_states, all_times_a, all_actions


def get_all_levels_df():
    subjects = []
    all_difficulties = []
    world = []
    all_states = []
    all_actions = []
    times_state = []
    times_action = []
    for subject in get_all_subjects():
        difficulties_s, world_s, times_s, states_s, times_a, actions_s = get_all_levels_for_subject(subject)
        all_difficulties.extend(difficulties_s)
        world.extend(world_s)
        times_state.extend(times_s)
        times_action.extend(times_a)
        all_states.extend(states_s)
        all_actions.extend(actions_s)
        subjects.append(subject)

    pd_dict = {'subject': subjects, 'difficulty': all_difficulties, 'world': world, 'times_states': times_state, 'states': all_states,
               'times_action': times_action, 'actions': all_actions}
    return pd.DataFrame.from_dict(pd_dict)


def assign_position_to_fields(x, y, width):
    field_y = int(round(y / config.FIELD_HEIGHT))
    field_x_start = int(round(x / config.FIELD_WIDTH))
    field_width = int(width // config.FIELD_WIDTH)
    return field_x_start, field_y, field_width

