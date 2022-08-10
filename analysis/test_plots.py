from matplotlib import pyplot as plt

from analysis.analysis_utils import get_eyetracker_samples
from analysis.plotting.gaze_plot import plot_gaze, plot_pupil_size_over_time
from game.world_generation.generation_config import GameDifficulty

# subject = 'TEST01'
# difficulty = GameDifficulty.EASY.value
# world_name = 'world_0'
#
# times_states = get_times_states(subject, difficulty, world_name)
# world_props = get_world_properties(subject, difficulty, world_name)
# target_position = int(world_props['target_position'])
#
# times, states = zip(*times_states)
#
# fig, ax = plt.subplots()
# discrete_plotting.plot_state(ax, states[0], target_position)
# plt.show()
#
# fm = create_feature_map_from_state(states[0], target_position)

subject = 'TADAAA'
difficulty = GameDifficulty.EASY.value
world_name = 'world_0'

samples = get_eyetracker_samples(subject, difficulty, world_name, training=True)

fig, ax = plt.subplots()
plot_gaze(ax, samples)
plt.show()

fig, ax = plt.subplots()
plot_pupil_size_over_time(ax, samples)
plt.show()

