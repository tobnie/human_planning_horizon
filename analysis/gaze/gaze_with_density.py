import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from analysis.analysis_utils import get_samples_for_all_players_in_row, \
    get_times_actions_states_samples_for_all, get_samples_from_time_state_action_samples, get_worlds_by_target_position
from analysis.gaze.gaze_plot import filter_off_samples


def joint_gaze_plot(gaze_points, xlim=(0, config.DISPLAY_WIDTH_PX), ylim=(0, config.DISPLAY_HEIGHT_PX)):
    x, y = zip(*gaze_points)
    g = sns.JointGrid(x=x, y=y, xlim=xlim, ylim=ylim)

    # adjust width of color bar
    pos_joint_ax = g.ax_joint.get_position()
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

    sns.kdeplot(x=x, y=y, shade=True, cmap="viridis", ax=g.ax_joint)
    sns.kdeplot(x=x, fill=True, ax=g.ax_marg_x, color="dimgrey")
    sns.kdeplot(y=y, fill=True, ax=g.ax_marg_y, color="dimgrey")

    # TODO do we want that?
    # # scatter gaze points
    # g.ax_joint.scatter(x, y, color='lightgray')

    g.fig.colorbar(g.ax_joint.collections[0], ax=[g.ax_joint, g.ax_marg_y, g.ax_marg_x], use_gridspec=True, orientation='horizontal')

    g.ax_joint.set_xticks(np.arange(xlim[0], xlim[1] + 1))
    g.ax_joint.set_yticks(np.arange(ylim[0], ylim[1] + 1))

    g.fig.suptitle('Gaze Point Density')
    g.fig.subplots_adjust(top=0.95, bottom=0.3)

    # g.fig.colorbar(g.ax_joint.collections[0], ax=[g.ax_joint, g.ax_marg_y, g.ax_marg_x], use_gridspec=True, orientation='horizontal')


def create_and_save_gaze_densities_per_row():
    for row in range(config.N_LANES):
        samples_row = get_samples_for_all_players_in_row(row=row)
        filtered_samples = filter_off_samples(samples_row)

        samples_coords_only = np.array([sample[1:-1] for sample in samples_row])

        # transform coordinates y
        samples_coords_only[:, 1] = config.DISPLAY_HEIGHT_PX - samples_coords_only[:, 1]

        joint_gaze_plot(samples_coords_only)
        plt.title('Row ' + str(row))
        plt.savefig(f'./imgs/gaze_per_row/gaze_density_row_{row}.png')
        plt.close(plt.gcf())


def create_and_save_gaze_density_for_all():
    t_a_a_s = get_times_actions_states_samples_for_all()
    samples = get_samples_from_time_state_action_samples(t_a_a_s)
    filtered_samples = filter_off_samples(samples)

    samples_coords_only = np.array(filtered_samples)[:, 1:-1]

    # transform coordinates y
    samples_coords_only[:, 1] = config.DISPLAY_HEIGHT_PX - samples_coords_only[:, 1]

    joint_gaze_plot(samples_coords_only)
    plt.savefig(f'./imgs/gaze_per_row/gaze_density.png')
    plt.close(plt.gcf())


def create_and_save_gaze_density_for_all_per_target():
    worlds_by_target_pos = get_worlds_by_target_position()
    tass = {'left': get_times_actions_states_samples_for_all(worlds=worlds_by_target_pos['left']),
            'center': get_times_actions_states_samples_for_all(worlds=worlds_by_target_pos['center']),
            'right': get_times_actions_states_samples_for_all(worlds=worlds_by_target_pos['right'])}

    for target_pos, t_a_a_s in tass.items():
        samples = get_samples_from_time_state_action_samples(t_a_a_s)
        filtered_samples = filter_off_samples(samples)

        samples_coords_only = np.array(filtered_samples)[:, 1:-1]

        # transform coordinates y
        samples_coords_only[:, 1] = config.DISPLAY_HEIGHT_PX - samples_coords_only[:, 1]

        joint_gaze_plot(samples_coords_only)
        plt.title('Target ' + target_pos)
        plt.savefig(f'./imgs/gaze_per_row/gaze_density_target_{target_pos}.png')
        plt.close(plt.gcf())


create_and_save_gaze_density_for_all_per_target()
