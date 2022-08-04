from matplotlib import pyplot as plt

from analysis.analysis_utils import get_times_states, get_world_properties, assign_position_to_fields, create_feature_map_from_state
import analysis.plotting_utils.fields as discrete_plotting
import analysis.plotting_utils.world_coordinates as cont_plotting
from world_generation.generation_config import GameDifficulty

subject = 'TEST01'
difficulty = GameDifficulty.EASY.value
world_name = 'world_0'

times_states = get_times_states(subject, difficulty, world_name)
world_props = get_world_properties(subject, difficulty, world_name)
target_position = int(world_props['target_position'])

times, states = zip(*times_states)

fig, ax = plt.subplots()
discrete_plotting.plot_state(ax, states[0], target_position)
plt.show()

fm = create_feature_map_from_state(states[0], target_position)
