from analysis.analysis_utils import get_times_states
from game.world_generation.generation_config import GameDifficulty

subject_id = 'KR07HA'

for difficulty in GameDifficulty:
    for i in range(20):
        times_states = get_times_states(subject_id, difficulty.value, 'world_{}'.format(i))
        times, states = list(zip(*times_states))

        # stack in np array


        # save as .npz