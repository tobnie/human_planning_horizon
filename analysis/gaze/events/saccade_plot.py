from matplotlib import pyplot as plt

import config
from analysis.analysis_utils import get_eyetracker_samples_only, get_all_subjects
from analysis.plotting.gaze.events.event_detection import get_saccades_from_samples
from analysis.world.world_coordinates import plot_world_background


def plot_saccades(ax, samples, title=''):
    # plot all events
    blinks = get_saccades_from_samples(samples)[1]  # [starttime, endtime, duration, startx, starty, endx, endy]

    for t_start, t_end, t, x_start, y_start, x_end, y_end in blinks:
        ax.plot([x_start, x_end], [config.DISPLAY_HEIGHT_PX - y_start, config.DISPLAY_HEIGHT_PX - y_end], c='orange')

    ax.set_xlim((0, config.DISPLAY_WIDTH_PX))
    ax.set_ylim((0, config.DISPLAY_HEIGHT_PX))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.tight_layout()


def plot_and_save_saccades():
    for subject in get_all_subjects():
        for difficulty in ['easy', 'normal', 'hard']:
            for i in range(20):
                fig, ax = plt.subplots()
                world_name = f'world_{i}'

                # get blinks
                samples = get_eyetracker_samples_only(subject, difficulty, world_name)

                # plotting
                plot_world_background(ax)
                plot_saccades(ax, samples, f'world {i}')
                plt.suptitle('Saccades for subject {} in difficulty {}'.format(subject, difficulty))
                plt.tight_layout()
                plt.savefig(f'./imgs/saccades/{subject}_{difficulty}_{i}_saccades.png')
                plt.close(fig)

