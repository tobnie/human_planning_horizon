import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import config
from analysis.analysis_utils import get_world_properties, get_eyetracker_samples_only, get_times_states, get_all_subjects
from analysis.plotting.plotting_utils import color_fader
from analysis.world.world_coordinates import plot_player_path
from game.world_generation.generation_config import GameDifficulty


def filter_off_samples(samples):
    if isinstance(samples, np.ndarray) and samples.size == 0:
        return []

    if isinstance(samples, list) and len(samples) == 0:
        return []

    samples = np.array(samples)
    return samples[(samples[:, 1] > -1000) & (samples[:, 2] > -1000) & (samples[:, 3] > 0)]


def get_times_from_samples(samples):
    return samples.T[0]


def get_gaze_coords_from_samples(samples):
    xy_coords = samples.T[1:3]
    return xy_coords[0], xy_coords[1]


def get_pupil_size_from_samples(samples):
    return samples.T[3]


def plot_gaze(ax, samples, color='orange'):
    coords = get_gaze_coords_from_samples(samples)
    coords = (coords[0], config.DISPLAY_HEIGHT_PX - coords[1])

    start_color = 'yellow'
    end_color = 'red'
    for i, coord in enumerate(zip(*coords)):
        # ax.plot(coord[0], coord[1], color=color_fader(start_color, end_color, i / len(coords[0])))
        ax.scatter(coord[0], coord[1], color=color_fader(start_color, end_color, i / len(coords[0])))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Game Duration: {} [ms]'.format(get_times_from_samples(samples)[-1]))
    ax.set_xlim(0, config.DISPLAY_WIDTH_PX)
    ax.set_ylim(0, config.DISPLAY_HEIGHT_PX)


def plot_pupil_size_over_time(ax, samples):
    times = get_times_from_samples(samples)
    pupil_size = get_pupil_size_from_samples(samples)

    ax.plot(times, pupil_size, color='black')
    ax.set_xlabel('time')
    ax.set_ylabel('pupil size [mm^2]')


def create_gaze_and_path_plots():


    for subject_id in get_all_subjects():
        print("Creating plots for subject:", subject_id)
        for y_subplot, difficulty_enum in enumerate(GameDifficulty):
            print('Difficulty:', difficulty_enum.value)
            for i in tqdm(range(20)):
                # meta properties
                subject = subject_id
                difficulty = difficulty_enum.value
                world_name = 'world_{}'.format(i)

                # get game data
                try:
                    samples = get_eyetracker_samples_only(subject, difficulty, world_name)
                    filtered_samples = filter_off_samples(samples)
                    times_states = get_times_states(subject, difficulty, world_name)
                    world_props = get_world_properties(subject, difficulty, world_name)
                    target_position = int(world_props['target_position'])
                    times, states = zip(*times_states)
                except FileNotFoundError:
                    continue

                # plot data
                fig, ax = plt.subplots()
                plt.suptitle('{} - World {}, {}'.format(subject_id, i, difficulty))
                plot_player_path(ax, times_states, target_position)
                plot_gaze(ax, filtered_samples)
                plt.tight_layout()
                plt.savefig('./imgs/gaze/{}_{}_world_{}.png'.format(subject_id, difficulty, i))
                plt.close(fig)
        print('Done!')

