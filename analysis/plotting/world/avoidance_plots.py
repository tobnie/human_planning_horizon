from matplotlib import pyplot as plt

from analysis.analysis_utils import get_times_actions_states, filter_times_actions_states_by_action, get_all_subjects, \
    get_all_times_actions_of_player
from analysis.plotting.actions.action_plots import plot_action_distribution
from analysis.plotting.world.feature_maps import get_avoidance_distribution_around_player_from_state_dict
from analysis.plotting.world.fields import plot_heatmap


def plot_avoidance_maps_by_actions_for_subject(subject_id, radius):
    time_action_states = get_times_actions_states(subject_id)
    sorted_by_action = filter_times_actions_states_by_action(time_action_states)
    avoidance_distributions = get_avoidance_distribution_around_player_from_state_dict(sorted_by_action, radius=radius)
    plot_avoidance_maps_by_actions(avoidance_distributions)
    side_length = 2 * radius + 1
    plt.suptitle(f'Avoidance maps for subject {subject_id} ({side_length}x{side_length})')


def plot_avoidance_maps_by_actions(action_avoidance_map_dict):
    fig, axes = plt.subplots(1, len(action_avoidance_map_dict.keys()), figsize=(12, 2))
    for i, (action, avoidance_map) in enumerate(action_avoidance_map_dict.items()):
        plot_heatmap(axes[i], avoidance_map, title=f'\'{action}\'')

        # set axis ticks
        map_diameter = avoidance_map.shape
        axes[i].set_xticklabels([int(label.get_text()) - map_diameter[0] // 2 for label in axes[i].get_xticklabels()])
        axes[i].set_yticklabels([int(label.get_text()) - map_diameter[1] // 2 for label in axes[i].get_yticklabels()])

    # TODO somehow looks strange. Change w.r.t. left/right direction please
    plt.tight_layout()


def plot_and_save_avoidance_maps():
    for radius in [1, 2, 3]:
        for subject in get_all_subjects():
            plot_avoidance_maps_by_actions_for_subject(subject, radius=radius)
            side_length = 2 * radius + 1
            plt.savefig('./imgs/avoidance_plots/{}_{}x{}_avoidance_map.png'.format(subject, side_length, side_length))
            plt.close(plt.gcf())

