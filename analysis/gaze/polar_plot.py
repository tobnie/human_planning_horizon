import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import config
from analysis.analysis_utils import get_times_states, get_eyetracker_samples_only, get_world_properties
from analysis.data_utils import get_all_subjects
from analysis.gaze.gaze_plot import filter_off_samples
from analysis.world.world_coordinates import get_center_of_player


def calc_angles_between_player_and_gaze(player_positions, gaze_positions):
    """Calculates the angle between the player and the 3gaze 2position"""
    # TODO maybe need to transform player and 3gaze y coordinates by HEIGHT - y?
    return np.arctan2(player_positions[:, 0] - gaze_positions[:, 0], player_positions[:, 1] - gaze_positions[:, 1])
    # return np.arctan2(gaze_positions[:, 1] - player_positions[:, 1], gaze_positions[:, 0] - player_positions[:, 0])


def polar_histogram(data, n_bins=90, with_kde=False):
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

    ax.set_theta_offset(np.pi / 2)


def plot_polar_gaze_angles(subject, difficulty, world_number):
    world_name = 'world_{}'.format(world_number)
    times_states = get_times_states(subject, difficulty, world_name)

    if not times_states:
        return
    times, states = zip(*times_states)

    player_pos = np.array([state[0][1:-1] for state in states])
    player_pos[:, 1] = config.DISPLAY_HEIGHT_PX - player_pos[:, 1] - config.PLAYER_HEIGHT
    player_pos = [get_center_of_player(x, y) for x, y in player_pos]

    samples = get_eyetracker_samples_only(subject, difficulty, world_name)
    samples = filter_off_samples(samples)
    if (isinstance(samples, np.ndarray) and samples.size == 0) or (isinstance(samples, list) and len(samples) == 0):
        return
    gaze_pos = samples[:, 1:3]

    # remove player positions where time does not match 3gaze 2position
    player_pos = np.array([player_pos[i] for i in range(len(player_pos)) if times[i] in samples[:, 0]])

    # TODO plot 3gaze angles from player relative to angle to target 2position
    world_props = get_world_properties(subject, difficulty, world_name)
    target_position = int(world_props['target_position'])

    gaze_angles = calc_angles_between_player_and_gaze(player_pos, gaze_pos)

    data = np.array([angle if angle >= 0 else angle + 2 * np.pi for angle in gaze_angles])
    polar_histogram(data, with_kde=False)

    plt.savefig('./imgs/gaze_polar/{}_{}_{}_polar_gaze.png'.format(subject, difficulty, world_number))
    plt.close(plt.gcf())


def plot_gaze_angles_for_all_subjects():
    # TODO check if angles are calculated correctly?
    for subject in get_all_subjects():
        for difficulty in ['easy', 'normal', 'hard']:
            for i in range(20):
                plot_polar_gaze_angles(subject, difficulty, i)
