from matplotlib import pyplot as plt

from analysis.analysis_utils import get_times_states, get_world_properties, create_feature_map_from_state, get_eyetracker_samples
import analysis.plotting_utils.fields as discrete_plotting
from analysis.plotting_utils.gaze_plot import plot_gaze
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
time = samples.T[0]
coords = samples.T[1:3]

print(time)
print(coords)

fig, ax = plt.subplots()
plot_gaze(ax, samples, coords)
plt.show()