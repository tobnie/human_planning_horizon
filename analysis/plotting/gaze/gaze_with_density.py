import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from analysis.analysis_utils import get_all_samples_for_subject
from analysis.plotting.gaze.gaze_plot import filter_off_samples


def joint_gaze_plot(gaze_points, xlim=(0, config.N_FIELDS_PER_LANE), ylim=(0, config.N_LANES)):
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


# TODO plot for all subjects and all gaze points
subject = 'KR07HA'
samples = get_all_samples_for_subject(subject)
filtered_samples = filter_off_samples(samples)

# remove time_stamps
samples_coords_only = np.array([sample[1:-1] for sample in samples])

# transform coordinates y
samples_coords_only[:, 1] = config.DISPLAY_HEIGHT_PX - samples_coords_only[:, 1]

random_samples = np.random.uniform(5, 10, size=(100, 2))

joint_gaze_plot(random_samples)
plt.show()
