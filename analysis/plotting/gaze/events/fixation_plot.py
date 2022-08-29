from matplotlib import pyplot as plt

import config
from analysis.analysis_utils import get_eyetracker_samples_only, get_all_subjects
from analysis.plotting.gaze.events.event_detection import get_fixations_from_samples
from analysis.plotting.world.world_coordinates import plot_world_background


def plot_fixation_times(ax, samples, title=''):
    # plot all events
    fixations = get_fixations_from_samples(samples)[1]  # [starttime, endtime, duration, endx, endy]

    for t_start, t_end, _, _, _ in fixations:
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


def plot_fixations(ax, samples, title=''):
    # plot all events
    fixations = get_fixations_from_samples(samples)[1]  # [starttime, endtime, duration, endx, endy]

    for t_start, t_end, t, x, y in fixations:
        plt.scatter(x, config.DISPLAY_HEIGHT_PX - y, marker="o", s=t, alpha=0.3, c='lightcoral')

    ax.set_xlim((0, config.DISPLAY_WIDTH_PX))
    ax.set_ylim((0, config.DISPLAY_HEIGHT_PX))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.tight_layout()


def plot_and_save_fixation_times():
    for subject in get_all_subjects():
        for difficulty in ['easy', 'normal', 'hard']:
            fig, ax = plt.subplots(4, 5)
            for i in range(20):
                world_name = f'world_{i}'

                # get blinks
                samples = get_eyetracker_samples_only(subject, difficulty, world_name)

                # plotting
                plot_fixation_times(ax[i // 5, i % 5], samples, f'world {i}')
            plt.suptitle('Fixations for subject {} in difficulty {}'.format(subject, difficulty))
            plt.tight_layout()
            plt.savefig(f'./imgs/fixations/{subject}_{difficulty}_fixation_times.png')
            plt.close(fig)


def plot_and_save_fixations():
    for subject in get_all_subjects():
        for difficulty in ['easy', 'normal', 'hard']:
            for i in range(20):
                fig, ax = plt.subplots()
                world_name = f'world_{i}'

                # get blinks
                samples = get_eyetracker_samples_only(subject, difficulty, world_name)

                # plotting
                plot_world_background(ax)
                plot_fixations(ax, samples, f'world {i}')
                plt.suptitle('Fixations for subject {} in difficulty {}'.format(subject, difficulty))
                plt.tight_layout()
                plt.savefig(f'./imgs/fixations/{subject}_{difficulty}_{i}_fixations.png')
                plt.close(fig)

