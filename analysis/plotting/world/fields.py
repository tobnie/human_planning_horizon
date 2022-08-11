import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import config
from analysis.analysis_utils import assign_position_to_fields
from analysis.plotting import plotting_utils
from analysis.plotting.plotting_utils import plot_rect

def plot_field(ax, x, y, width, color='k'):
    plot_rect(ax, x, config.N_LANES - y - 1, width, 1,
              color=color)


def plot_street_fields(ax):
    street_width = config.N_FIELDS_PER_LANE
    street_height = config.N_STREET_LANES
    street_y = 1
    plot_rect(ax, 0, street_y, street_width, street_height, color='gray')


def plot_water_fields(ax):
    water_width = config.N_FIELDS_PER_LANE
    water_height = config.N_WATER_LANES
    water_y = config.N_STREET_LANES + 2
    plot_rect(ax, 0, water_y, water_width, water_height, color='lightblue')


def plot_target_position_fields(ax, target_position):
    plot_rect(ax, target_position, config.N_LANES - 1, 1,
              config.ROW_HEIGHT, 'yellow')


def plot_world_background_fields(ax, target_position):
    plot_target_position_fields(ax, target_position)
    plot_water_fields(ax)
    plot_street_fields(ax)


def draw_grid(ax):
    plotting_utils.draw_grid(ax, 1, 1)


def plot_state(ax, state, target_position, time=None):
    """ Plots a representation of the given state."""

    # plot background
    plot_world_background_fields(ax, target_position)

    # plot entities
    for entry in reversed(state):  # reverse it so player will be drawn last and ON lilypads, not below them
        field = assign_position_to_fields(*entry[1:])
        if entry[0] == 0:
            color = 'purple'
            field = field[0], field[1], 1
        elif entry[0] == 1:
            color = 'red'
        elif entry[0] == 2:
            color = 'green'
        else:
            raise RuntimeError(f"Unknown entity encoding in .npz-file: {entry[0]}")

        plot_field(ax, *field, color=color)

    # plot properties
    if time:
        plt.title(f't={time}')

    draw_grid(ax)
    plt.xlim((0, config.N_FIELDS_PER_LANE))
    plt.ylim((0, config.N_LANES))
    plt.xticks(np.arange(0, config.N_FIELDS_PER_LANE, 1))
    plt.yticks(np.arange(0, config.N_LANES, 1))


def plot_heatmap(ax, data, title='Heatmap'):
    """ Plots a discrete heatmap over all fields of the world. """
    sns.heatmap(data, ax=ax)
    ax.set_title(title)
