import csv
import os

import numpy as np
from tqdm import tqdm

import config
from itertools import product
import pandas as pd

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
    return do_something_for_all_worlds(subject, get_eyetracker_samples, worlds)


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

    if not times_actions or not times_states:
        return []
    times_actions_states = []

    action_index = 0
    for t_s, state in times_states:

        next_t_a = times_actions[action_index + 1][0] if action_index + 1 < len(times_actions) else np.inf
        action = times_actions[action_index][1]

        if t_s < next_t_a:
            times_actions_states.append((t_s, action, state))
        else:
            action_index += 1
            action = times_actions[action_index][1]
            times_actions_states.append((t_s, action, state))

    return times_actions_states


def get_all_subjects():
    data_dir = f'../data/level_data/'
    subjects = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir + f)]
    return subjects


def get_target_position_for_world(diff, world):
    dummy_subject = get_all_subjects()[0]
    world_properties = get_world_properties(dummy_subject, diff, world)
    return int(world_properties['target_position'])


def get_times_actions_states_samples(subject_id, diffs_and_worlds=None):
    """ Returns all samples for each state as list of [(time, state, [samples])]."""
    times_action_states = get_times_actions_states(subject_id, diffs_and_worlds)
    if not times_action_states:
        return [(np.nan, np.nan, np.nan, [(np.nan, np.nan, np.nan, np.nan)])]

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


def field2screen(field_coords):
    FIELD_WIDTH = config.FIELD_WIDTH
    FIELD_HEIGHT = config.FIELD_HEIGHT
    screen_coords = [(x * FIELD_WIDTH + FIELD_WIDTH / 2, y * FIELD_HEIGHT + FIELD_HEIGHT / 2) for x, y in field_coords]
    return np.array(screen_coords)


def assign_position_to_fields(x, y, width):
    field_y = int(round(y / config.FIELD_HEIGHT))
    field_x_start = int(round(x / config.FIELD_WIDTH))
    field_width = int(width // config.FIELD_WIDTH)
    return field_x_start, field_y, field_width


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


def run_preprocessing():
    # we want to save each row with:
    # subject_id, game_difficulty, world_number, time, gaze_x, gaze_y, pupil_size, player_x, player_y, action, state, (?)

    # TODO information on left / right moving lane?

    dict_to_save = {
        'subject_id': [],
        'game_difficulty': [],
        'world_number': [],
        'target_position': [],
        'time': [],
        'gaze_x': [],
        'gaze_y': [],
        'pupil_size': [],
        'player_x': [],
        'player_y': [],
        'action': [],
        'state': [],
        # 'state_fm': []
    }

    subject_ids = []
    game_difficulties = []
    world_numbers = []
    times = []
    gaze_xs = []
    gaze_ys = []
    pupil_sizes = []
    player_xs = []
    player_ys = []
    actions = []
    target_positions = []
    states = []
    # state_fms = []

    # do for all subjects
    difficulties = ['easy', 'normal', 'hard']
    world_range = range(20)
    # sub_diff_worlds = list(product(['KR07HA'], difficulties, world_range))
    sub_diff_worlds = list(product(get_all_subjects(), difficulties, world_range))

    saved_first_subject = False
    print('Starting preprocessing...')
    for subject_id, diff, world_nr in tqdm(sub_diff_worlds):
        target_position = get_target_position_for_world(diff, f'world_{world_nr}')

        tass = get_times_actions_states_samples(subject_id, [(diff, f'world_{world_nr}')])
        for time, action, state, samples in tass:
            for sample in samples:
                # id
                subject_ids.append(subject_id)

                # difficulty
                game_difficulties.append(diff)

                # world number
                world_numbers.append(world_nr)

                # time
                times.append(sample[0])

                # eye samples
                gaze_xs.append(sample[1])
                # TODO gaze transformation correct?
                gaze_y = config.DISPLAY_HEIGHT_PX - sample[2]
                gaze_ys.append(gaze_y)
                pupil_sizes.append(sample[3])

                # target position
                target_position_screen_coords = config.FIELD_WIDTH * target_position + config.FIELD_WIDTH / 2
                target_positions.append(target_position_screen_coords)

                # action
                if np.isnan(action):
                    actions.append(np.nan)
                else:
                    actions.append(ACTION_TO_STRING[action])

                # state
                if (isinstance(state, np.ndarray) and state.size == 0) or (isinstance(state, list) and len(state) == 0) or (
                        isinstance(state, float) and np.isnan(state)):
                    states.append(np.nan)
                    # state_fms.append(np.nan)
                    player_xs.append(np.nan)
                    player_ys.append(np.nan)
                else:
                    state_as_list = state.tolist()
                    states.append(state_as_list)

                    # fm
                    # state_fm = create_feature_map_from_state(state)
                    # state_fms.append(state_fm)

                    # player position
                    player_pos = state[0, 1:3]
                    player_width = state[0, 3]
                    player_height = config.PLAYER_HEIGHT
                    player_x = player_pos[0] + player_width / 2
                    player_y = config.DISPLAY_HEIGHT_PX - player_pos[1] - player_height / 2
                    player_xs.append(player_x)
                    player_ys.append(player_y)

        if not saved_first_subject:
            subject_dict = {
                'subject_id': subject_ids,
                'game_difficulty': game_difficulties,
                'world_number': world_numbers,
                'target_position': target_positions,
                'time': times,
                'gaze_x': gaze_xs,
                'gaze_y': gaze_ys,
                'pupil_size': pupil_sizes,
                'player_x': player_xs,
                'player_y': player_ys,
                'action': actions,
                'state': states
            }

            subject_df = pd.DataFrame(subject_dict)

            subject_df.to_csv(f'../data/{subject_id}_compressed.gzip', compression='gzip')
            saved_first_subject = True

    dict_to_save['subject_id'] = subject_ids
    dict_to_save['game_difficulty'] = game_difficulties
    dict_to_save['world_number'] = world_numbers
    dict_to_save['target_position'] = target_positions
    dict_to_save['time'] = times
    dict_to_save['gaze_x'] = gaze_xs
    dict_to_save['gaze_y'] = gaze_ys
    dict_to_save['pupil_size'] = pupil_sizes
    dict_to_save['player_x'] = player_xs
    dict_to_save['player_y'] = player_ys
    dict_to_save['action'] = actions
    dict_to_save['state'] = states
    # dict_to_save['state_fm'] = state_fms

    # save dataframe
    df = pd.DataFrame(dict_to_save)

    df.to_csv('../data/data_compressed.gzip', compression='gzip')
    print('done')


run_preprocessing()
