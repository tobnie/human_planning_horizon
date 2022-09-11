import csv
import os

import numpy as np
from tqdm import tqdm

import config
from itertools import product
import pandas as pd

from analysis.data_utils import get_all_subjects, assign_position_to_fields, add_game_status_to_df
from analysis.player.player_position_heatmap import add_player_position_in_field_coordinates
from analysis.sosci_utils import add_experience_to_df
from analysis.score_utils import add_max_score_to_df
from analysis.trial_order_utils import add_trial_numbers_to_df
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

    for state in states:
        # transform
        # margin_between_player_and_field_y = (config.FIELD_HEIGHT - config.PLAYER_HEIGHT) / 2
        state[0, 2] = config.DISPLAY_HEIGHT_PX - state[0, 2] - config.PLAYER_HEIGHT

        # transform y of each state:
        state[1:, 2] = config.DISPLAY_HEIGHT_PX - state[1:, 2] - config.FIELD_HEIGHT

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

        t_a = times_actions[action_index + 1][0] if action_index + 1 < len(times_actions) else np.inf

        if t_s == t_a:
            action = times_actions[action_index][1]
            times_actions_states.append((t_s, action, state))
            action_index += 1
        else:
            times_actions_states.append((t_s, np.nan, state))

    return times_actions_states


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

    return feature_map


def run_preprocessing():
    # we want to save each row with:
    # subject_id, game_difficulty, world_number, time, gaze_x, gaze_y, pupil_size, player_x, player_y, action, state, (?)

    # TODO information on left / right moving lane?

    # do for all subjects
    difficulties = ['easy', 'normal', 'hard']
    world_range = range(20)

    print('Starting preprocessing...')
    for subject_id in get_all_subjects():
        print(f'Subject {subject_id} next.')

        subject_ids_i = []
        game_difficulties_i = []
        world_numbers_i = []
        times_i = []
        gaze_xs_i = []
        gaze_ys_i = []
        pupil_sizes_i = []
        player_xs_i = []
        player_ys_i = []
        actions_i = []
        target_positions_i = []
        states_i = []

        for diff, world_nr in tqdm(list(product(difficulties, world_range))):
            target_position = get_target_position_for_world(diff, f'world_{world_nr}')

            tass = get_times_actions_states_samples(subject_id, [(diff, f'world_{world_nr}')])
            for time, action, state, samples in tass:
                for sample in samples:
                    # id
                    subject_ids_i.append(subject_id)

                    # difficulty
                    game_difficulties_i.append(diff)

                    # world number
                    world_numbers_i.append(world_nr)

                    # time
                    times_i.append(sample[0])

                    # eye samples
                    gaze_xs_i.append(sample[1])
                    # TODO gaze transformation correct?
                    gaze_y = config.DISPLAY_HEIGHT_PX - sample[2]
                    gaze_ys_i.append(gaze_y)
                    pupil_sizes_i.append(sample[3])

                    # target position
                    target_position_screen_coords = config.FIELD_WIDTH * target_position + config.FIELD_WIDTH / 2
                    target_positions_i.append(target_position_screen_coords)

                    # action
                    if np.isnan(action):
                        actions_i.append(np.nan)
                    else:
                        actions_i.append(ACTION_TO_STRING[action])

                    # state
                    if (isinstance(state, np.ndarray) and state.size == 0) or (isinstance(state, list) and len(state) == 0) or (
                            isinstance(state, float) and np.isnan(state)):
                        states_i.append(np.nan)
                        # state_fms.append(np.nan)
                        player_xs_i.append(np.nan)
                        player_ys_i.append(np.nan)
                    else:
                        state_as_list = state.tolist()
                        states_i.append(state_as_list)

                        # fm
                        # state_fm = create_feature_map_from_state(state)
                        # state_fms.append(state_fm)

                        # player position
                        player_pos = state[0, 1:3]
                        player_width = state[0, 3]
                        player_height = config.PLAYER_HEIGHT
                        player_x = player_pos[0] + player_width / 2
                        player_y = player_pos[1] + player_height / 2
                        player_xs_i.append(player_x)
                        player_ys_i.append(player_y)

        subject_dict = {
            'subject_id': subject_ids_i,
            'game_difficulty': game_difficulties_i,
            'world_number': world_numbers_i,
            'target_position': target_positions_i,
            'time': times_i,
            'gaze_x': gaze_xs_i,
            'gaze_y': gaze_ys_i,
            'pupil_size': pupil_sizes_i,
            'player_x': player_xs_i,
            'player_y': player_ys_i,
            'action': actions_i,
            'state': states_i
        }

        subject_df = pd.DataFrame(subject_dict)

        # missing y samples get transformed as well, so replace them again to be classified as missing correctly.
        subject_df = subject_df.replace([34208], -32768.0)

        subject_df = add_game_status_to_df(subject_df)
        subject_df = add_player_position_in_field_coordinates(subject_df)
        subject_df = add_experience_to_df(subject_df)
        subject_df = add_trial_numbers_to_df(subject_df)
        subject_df = add_max_score_to_df(subject_df)

        subject_df.to_csv(f'../data/compressed_data/{subject_id}_compressed.gzip', compression='gzip')

    print('done')
