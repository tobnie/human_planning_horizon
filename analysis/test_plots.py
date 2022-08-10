from matplotlib import pyplot as plt

from analysis.analysis_utils import get_eyetracker_samples, get_eyetracker_events, get_times_states, get_world_properties
from analysis.plotting.gaze.event_plot import plot_blinks
from analysis.plotting.gaze.gaze_event_utils import get_blinks
from analysis.plotting.gaze.gaze_plot import plot_gaze, plot_pupil_size_over_time, filter_off_samples
from analysis.plotting.world.world_coordinates import plot_player_path
from game.world_generation.generation_config import GameDifficulty

subject = 'KR07HA'
difficulty = GameDifficulty.EASY.value
world_name = 'world_2'

samples = get_eyetracker_samples(subject, difficulty, world_name)
filtered_samples = filter_off_samples(samples)

times_states = get_times_states(subject, difficulty, world_name)
world_props = get_world_properties(subject, difficulty, world_name)
target_position = int(world_props['target_position'])

times, states = zip(*times_states)

fig, ax = plt.subplots()

plot_player_path(ax, times_states, target_position)
plot_gaze(ax, filtered_samples)

plt.savefig('test.png')
plt.show()

# fig, ax = plt.subplots()
# plot_pupil_size_over_time(ax, samples)
# plt.show()
#

# events = get_eyetracker_events(subject, difficulty, world_name)
#
# blink_starts, blink_ends = get_blinks(events)
# print(blink_starts)
# print(blink_ends)
# fig, ax = plt.subplots()
# plot_blinks(ax, times, blink_starts, blink_ends)
# plt.show()
