import os

import numpy as np
from tqdm import tqdm

from analysis.analysis_utils import get_times_states, get_level_data_path
from game.world_generation.generation_config import GameDifficulty


# TODO also do for training data
def transform_subject_data(s_id_list):
    for subject_id in s_id_list:
        for difficulty in GameDifficulty:
            for i in tqdm(range(20)):
                world_name = 'world_{}'.format(i)
                log_directory = get_level_data_path(subject_id, difficulty.value, world_name, False)

                times_states = get_times_states(subject_id, difficulty.value, world_name)
                if times_states:
                    times, states = list(zip(*times_states))
                    times = np.array(times)
                    states = np.array(states)

                    # save as .npz
                    npz_dict = {'times': times, 'states': states}
                    np.savez_compressed(log_directory + f'world_states.npz', **npz_dict)


def remove_old_data():
    base_dir = './level_data/'
    for subject in os.listdir(base_dir):
        if os.path.isdir(base_dir + subject):
            difficulty_folders = os.listdir(base_dir + subject)
            for difficulty_folder in difficulty_folders:
                if os.path.isdir(base_dir + subject + '/' + difficulty_folder):
                    for i in range(20):
                        world_name = 'world_{}'.format(i)

                        world_directory = base_dir + subject + '/' + difficulty_folder + '/' + world_name + '/'
                        # get states_files in log_directory
                        states_directory = world_directory + 'states/'
                        if os.path.exists(states_directory):
                            states_files = [f for f in os.listdir(states_directory) if f.endswith('.npz')]
                            for states_file in states_files:
                                os.remove(states_directory + states_file)
                            os.removedirs(states_directory)


transform_subject_data(['ED06RA'])
remove_old_data()
