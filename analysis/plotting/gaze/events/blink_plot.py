from matplotlib import pyplot as plt

import config
from analysis.analysis_utils import get_eyetracker_samples, get_all_subjects
from analysis.plotting.gaze.events.event_detection import get_blinks_from_samples


def plot_blinks(ax, samples, title=''):
    # plot all events
    blinks = get_blinks_from_samples(samples)[1]

    for t_start, t_end, b_duration in blinks:
        ax.axvspan(t_start, t_end, alpha=0.5, color='r')

    ax.set_xlim((0, config.LEVEL_TIME))
    # ax.set_xlabel('time')
    # ax.set_ylabel('events')
    ax.set_title(title)
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.tight_layout()


def plot_and_save_blinks():
    for subject in get_all_subjects():
        for difficulty in ['easy', 'normal', 'hard']:
            fig, ax = plt.subplots(4, 5)
            for i in range(20):
                world_name = f'world_{i}'

                # get blinks
                samples = get_eyetracker_samples(subject, difficulty, world_name)

                # plotting
                plot_blinks(ax[i // 5, i % 5], samples, f'world {i}')
            plt.suptitle('Blinks for subject {} in difficulty {}'.format(subject, difficulty))
            plt.tight_layout()
            plt.savefig(f'./imgs/blinks/{subject}_{difficulty}_blinks.png')


plot_and_save_blinks()
