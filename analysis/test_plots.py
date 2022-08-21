from matplotlib import pyplot as plt

from analysis.analysis_utils import get_world_properties
from analysis.plotting.world.avoidance_plots import plot_avoidance_maps_by_actions_for_subject
from game.world_generation.generation_config import GameDifficulty

subject = 'KR07HA'
difficulty = GameDifficulty.EASY.value
world_name = 'world_12'
world_props = get_world_properties(subject, difficulty, world_name)

plot_avoidance_maps_by_actions_for_subject(subject)
plt.show()
# states = get_states(subject, difficulty, world_name)
# feature_maps = states_to_feature_maps(states)
# feature_maps_around_player = get_area_around_player(feature_maps)
# avoidance_fm_3x3 = get_feature_map_distribution_for_avoidance(feature_maps_around_player)
#
# fig, ax = plt.subplots()
# plot_heatmap(ax, avoidance_fm_3x3, title='Avoidance feature map 3x3 around player')
# plt.show()
#
# # only get player states
# summed_player_fm = get_feature_map_distribution_for_player(feature_maps)
#
# area_around_player = get_area_around_player(feature_maps[0])
#
# fig, ax = plt.subplots()
# plot_heatmap(ax, summed_player_fm)
# plt.show()

# subject = 'KR07HA'
# difficulty = GameDifficulty.EASY.value
# world_name = 'world_0'
#
# all_difficulties, world, times_s, all_states, times_a, all_actions = get_all_levels_for_subject(subject)
# print(len(all_difficulties))
# print(len(world))
# print(len(times_s))
# print(len(all_states))
# print(len(times_a))
# print(len(all_actions))

# # samples = get_eyetracker_samples(subject, difficulty, world_name, training=True)
# # filtered_samples = filter_off_samples(samples)
# times_states = get_times_states(subject, difficulty, world_name, training=True)
# world_props = get_world_properties(subject, difficulty, world_name, training=True)
# target_position = int(world_props['target_position'])
#
# fig, ax = plt.subplots()
# plot_state(ax, times_states[10][1], target_position)
# # plot_player_path(ax, times_states, target_position)
# plt.savefig('test.png')
#
# plt.show()

# subject = 'TI01NI'
# difficulty = GameDifficulty.NORMAL.value
# world_name = 'world_1'
#
# samples = get_eyetracker_samples(subject, difficulty, world_name, training=True)
# filtered_samples = filter_off_samples(samples)
# times_states = get_times_states(subject, difficulty, world_name, training=True)
# world_props = get_world_properties(subject, difficulty, world_name, training=True)
# target_position = int(world_props['target_position'])
# times, states = zip(*times_states)
#
# fig, ax = plt.subplots()
#
# plot_player_path(ax, times_states, target_position)
# plot_gaze(ax, filtered_samples)
#
# plt.savefig('test.png')
# plt.show()

#
# def create_plots_for_subject_ids(subject_ids):
#     for subject_id in subject_ids:
#         print("Creating plots for subject:", subject_id)
#         for y_subplot, difficulty_enum in enumerate(GameDifficulty):
#             print('Difficulty:', difficulty_enum.value)
#             for i in tqdm(range(20)):
#                 # meta properties
#                 subject = subject_id
#                 difficulty = difficulty_enum.value
#                 world_name = 'world_{}'.format(i)
#
#                 # get game data
#                 try:
#                     samples = get_eyetracker_samples(subject, difficulty, world_name)
#                     filtered_samples = filter_off_samples(samples)
#                     times_states = get_times_states(subject, difficulty, world_name)
#                     world_props = get_world_properties(subject, difficulty, world_name)
#                     target_position = int(world_props['target_position'])
#                     times, states = zip(*times_states)
#                 except FileNotFoundError:
#                     continue
#
#                 # plot data
#                 fig, ax = plt.subplots()
#                 plt.suptitle('{} - World {}, {}'.format(subject_id, i, difficulty))
#                 plot_player_path(ax, times_states, target_position)
#                 plot_gaze(ax, filtered_samples)
#                 plt.tight_layout()
#                 plt.savefig('imgs/gaze/{}_{}_world_{}.png'.format(subject_id, difficulty, i))
#         print('Done!')
#
# create_plots_for_subject_ids(['TI01NI'])


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
