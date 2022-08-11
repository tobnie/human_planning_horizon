import numpy as np
from tqdm import tqdm

from analysis.analysis_utils import get_times_states, get_level_data_path
from game.world_generation.generation_config import GameDifficulty

s_id_list = ['KR07HA', 'NI07LU', 'TI01NI']

for subject_id in s_id_list:
    for difficulty in GameDifficulty:
        for i in tqdm(range(20)):
            world_name = 'world_{}'.format(i)
            log_directory = get_level_data_path(subject_id, difficulty.value, world_name, False)

            times_states = get_times_states(subject_id, difficulty.value, world_name)
            times, states = list(zip(*times_states))
            times = np.array(times)
            states = np.array(states)

            # save as .npz
            npz_dict = {'times': times, 'states': states}
            np.savez_compressed(log_directory + f'world_states.npz', **npz_dict)
