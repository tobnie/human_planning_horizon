import csv
import os
from itertools import product

import numpy as np
import pandas as pd

import config

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


def do_something_for_all_worlds(subject_id, func, diffs_and_worlds=None):
    result_acc = []

    if subject_id == 'dummy':
        subject_id = get_all_subjects()[0]

    if not diffs_and_worlds:
        diffs_and_worlds = list(product(GameDifficulty, [f'world_{i}' for i in range(20)]))

    for diff, world in diffs_and_worlds:
        diff = diff if isinstance(diff, str) else diff.value
        single_result = func(subject_id, diff, world)
        result_acc.extend(single_result)
    return result_acc


def get_all_samples_for_subject(subject, worlds=None):
    return do_something_for_all_worlds(subject, get_eyetracker_samples_only, worlds)


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


def get_eyetracker_samples_only(subject, difficulty, world_name, training=False):
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


def get_all_times_states_of_player(subject_id, diffs_and_worlds=None):
    return do_something_for_all_worlds(subject_id, get_times_states, diffs_and_worlds)


def get_all_times_actions_of_player(subject_id, diffs_and_worlds=None):
    return do_something_for_all_worlds(subject_id, get_times_actions, diffs_and_worlds)


def get_times_actions_states(subject_id, diffs_and_worlds=None):
    times_actions = get_all_times_actions_of_player(subject_id, diffs_and_worlds)
    times_states = get_all_times_states_of_player(subject_id, diffs_and_worlds)

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


def get_times_actions_states_samples_for_all(subjects=None, worlds=None):
    all_t_a_s_s = []

    subject_range = get_all_subjects() if subjects is None else subjects

    for subject in subject_range:
        t_a_s_s = get_times_actions_states_samples(subject, worlds)
        all_t_a_s_s.extend(t_a_s_s)
    return all_t_a_s_s


def get_target_position_for_world(diff, world):
    dummy_subject = get_all_subjects()[0]
    world_properties = get_world_properties(dummy_subject, diff, world)
    return int(world_properties['target_position'])


def return_world_if_target_position_left(subject_id, difficulty, world):
    target_position = get_target_position_for_world(difficulty, world)

    if target_position == 3:
        return [(difficulty, world)]
    else:
        return []


def return_world_if_target_position_center(subject_id, difficulty, world):
    target_position = get_target_position_for_world(difficulty, world)

    if target_position == 9:
        return [(difficulty, world)]
    else:
        return []


def return_world_if_target_position_right(subject_id, difficulty, world):
    target_position = get_target_position_for_world(difficulty, world)

    if target_position == 16:
        return [(difficulty, world)]
    else:
        return []


def get_worlds_by_target_position():
    target_left = list(filter(None, do_something_for_all_worlds('dummy', return_world_if_target_position_left, None)))
    target_center = list(filter(None, do_something_for_all_worlds('dummy', return_world_if_target_position_center, None)))
    target_right = list(filter(None, do_something_for_all_worlds('dummy', return_world_if_target_position_right, None)))
    return {'left': target_left, 'center': target_center, 'right': target_right}


def get_times_actions_states_samples_per_target_position_for_all():
    # get world for target positions
    worlds_per_target_position = get_worlds_by_target_position()

    t_a_s_s_per_target = {
        'left': [],
        'center': [],
        'right': []
    }
    for position_string, worlds in worlds_per_target_position.items():
        all_t_a_s_s = []
        for subject in get_all_subjects():
            t_a_s_s = get_times_actions_states_samples(subject, worlds)
            all_t_a_s_s.extend(t_a_s_s)
        t_a_s_s_per_target[position_string] = all_t_a_s_s
    return t_a_s_s_per_target


def get_times_actions_states_samples(subject_id, diffs_and_worlds=None):
    """ Returns all samples for each state as list of [(time, state, [samples])]."""
    times_action_states = get_times_actions_states(subject_id, diffs_and_worlds)
    samples = get_all_samples_for_subject(subject_id, diffs_and_worlds)

    # times actions states samples
    t_a_s_s = []
    samples_idx = 0
    for i in range(len(times_action_states) - 1):
        samples_for_state = []
        time, action, state = times_action_states[i]
        next_time = times_action_states[i + 1][0]

        # collect all samples with time bigger than current state time and smaller than next state time
        if samples_idx < len(samples):
            if next_time > time:
                while samples[samples_idx][0] < next_time:
                    samples_for_state.append(samples[samples_idx])
                    samples_idx += 1
            else:
                while samples[samples_idx][0] > next_time:
                    samples_for_state.append(samples[samples_idx])
                    samples_idx += 1

        t_a_s_s.append((time, action, state, samples_for_state))
    return t_a_s_s


def filter_times_actions_states_and_samples_for_row(time_action_state_samples, row, return_fms=True):
    """ Returns only time-action-fm-pairs and samples where the player is in the given row"""
    filtered_indices = []
    time_action_fms_samples = transform_times_actions_states_samples_to_fm(time_action_state_samples)
    times, actions, fms, samples = zip(*time_action_fms_samples)
    for i, fm in enumerate(fms):
        if fm[row, :, 0].sum() > 0:
            filtered_indices.append(i)

    if return_fms:
        return [time_action_fms_samples[i] for i in filtered_indices]
    else:
        return [time_action_state_samples[i] for i in filtered_indices]


def get_samples_from_time_state_action_samples(time_action_state_samples):
    """ Returns all samples from the given time-action-state-samples"""
    all_samples = []

    for time_action_state_sample in time_action_state_samples:
        sample = time_action_state_sample[-1]
        all_samples.extend(sample)
    return all_samples


def field2screen(field_coords):
    FIELD_WIDTH = config.FIELD_WIDTH
    FIELD_HEIGHT = config.FIELD_HEIGHT
    screen_coords = [(x * FIELD_WIDTH + FIELD_WIDTH / 2, y * FIELD_HEIGHT + FIELD_HEIGHT / 2) for x, y in field_coords]
    return np.array(screen_coords)


def get_times_actions_states_samples_for_row(subject_id, row, worlds=None):
    """ Returns all time-action-state-samples when the player is in a specific row"""
    time_action_state_samples = get_times_actions_states_samples(subject_id, worlds)
    filtered_time_action_state_samples = filter_times_actions_states_and_samples_for_row(time_action_state_samples, row=row)
    return filtered_time_action_state_samples


def get_times_actions_states_samples_for_all_players_in_row(row, worlds=None):
    """ Returns all time-action-state-samples when the player is in a specific row"""
    all_tass = []
    for subject_id in get_all_subjects():
        tass_subject = get_times_actions_states_samples_for_row(subject_id, row, worlds)
        all_tass.extend(tass_subject)
    return all_tass


def get_samples_for_player_in_row(subject_id, row):
    """ Returns all samples when the player is in a specific row"""
    # TODO filter samples here already?
    filtered_time_action_state_samples = get_times_actions_states_samples_for_row(subject_id, row)
    all_filtered_samples = get_samples_from_time_state_action_samples(filtered_time_action_state_samples)
    return all_filtered_samples


def get_samples_for_all_players_in_row(row):
    """ Returns all samples when the player is in a specific row"""
    all_samples = []
    for subject in get_all_subjects():
        subject_samples = get_samples_for_player_in_row(subject, row)
        all_samples.extend(subject_samples)
    return all_samples


def transform_times_actions_states_samples_to_fm(time_action_state_samples):
    """ Returns only time-action-fm-pairs and samples where the player is in the given row"""
    times, actions, states, samples = zip(*time_action_state_samples)
    fms = states_to_feature_maps(states)
    return list(zip(times, actions, fms, samples))


def get_player_position_from_state(state):
    """ Returns the position of the player in the given state"""
    return state[:, :, 0]


def assign_position_to_fields(x, y, width):
    field_y = int(round(y / config.FIELD_HEIGHT))
    field_x_start = int(round(x / config.FIELD_WIDTH))
    field_width = int(width // config.FIELD_WIDTH)
    return field_x_start, field_y, field_width


def states_to_feature_maps(list_of_states):
    """ Transforms a list of states into an array of feature maps. States are distributed along axis 0.
     Feature Maps have the following form: ['state', 'x', 'y', 'type']
     Types are Player: 0, Vehicle: 1, LilyPad: 2"""
    return np.array([create_feature_map_from_state(state) for state in list_of_states])


def create_feature_map_from_state(state):
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES, 3))

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

        feature_map[x_start:x_start + width, y, obj_type] = 1

    feature_map = np.rot90(feature_map)

    # invert y axis
    feature_map = np.flip(feature_map, axis=1)

    return feature_map

# worlds_by_target_pos = get_worlds_by_target_position()
# tass_left = get_times_actions_states_samples_for_all(worlds=worlds_by_target_pos['left'])
# tass_center = get_times_actions_states_samples_for_all(worlds=worlds_by_target_pos['center'])
# tass_right = get_times_actions_states_samples_for_all(worlds=worlds_by_target_pos['right'])
