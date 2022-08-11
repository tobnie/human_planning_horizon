import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def calc_angle_between_player_and_gaze(player_pos, gaze_pos):
    """Calculates the angle between the player and the gaze position"""
    return np.arctan2(gaze_pos[1] - player_pos[1], gaze_pos[0] - player_pos[0])


def polar_histogram(data, n_bins=360, with_kde=False):
    """Takes data in radians and plots a polar histogram"""

    # number of equal bins
    bins = np.linspace(0.0, 2 * np.pi, n_bins + 1)
    n, _, _ = plt.hist(data, bins, density=True)

    plt.clf()
    width = 2 * np.pi / n_bins
    ax = plt.subplot(1, 1, 1, projection='polar')
    bars = ax.bar(bins[:n_bins], n, width=width, bottom=0.0)
    for bar in bars:
        bar.set_alpha(0.5)

    if with_kde:
        # TODO how to do kde in polar coordinates? transform to cartesian space and do kde there?
        cartesian_x = np.cos(data)
        cartesian_y = np.sin(data)

        cartesian_coords = np.vstack([cartesian_x, cartesian_y])
        kde = stats.gaussian_kde(cartesian_coords)
        y = kde(cartesian_coords)

        x = np.linspace(0, 2 * np.pi, 1000)
        plt.plot(x, y)

    plt.show()


data = np.random.uniform(0, 2 * np.pi, size=100_000)
data = np.random.normal(0, np.pi / 2, size=100_000)
data = data % (2 * np.pi)
polar_histogram(data, with_kde=True)
