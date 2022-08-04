from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

import config


def plot_rect(ax, x, y, width, height, color):
    rect = Rectangle((x, y), width, height, color=color)
    ax.add_patch(rect)


def plot_target_position(ax, target_position):
    plot_rect(ax, target_position * config.FIELD_WIDTH, (config.N_LANES - 1) * config.ROW_HEIGHT, config.FIELD_WIDTH,
              config.ROW_HEIGHT, 'yellow')


def plot_water(ax):
    water_width = config.DISPLAY_WIDTH_PX
    water_height = config.N_WATER_LANES * config.ROW_HEIGHT
    water_y = config.DISPLAY_HEIGHT_PX - (config.ROW_HEIGHT + water_height) + 1
    plot_rect(ax, 0, water_y, water_width, water_height, color='lightblue')


def plot_street(ax):
    street_width = config.DISPLAY_WIDTH_PX
    street_height = config.N_STREET_LANES * config.ROW_HEIGHT
    street_y = config.ROW_HEIGHT
    plot_rect(ax, 0, street_y, street_width, street_height, color='gray')


def plot_lilypad(ax, x, y, width):
    plot_rect(ax, x, config.DISPLAY_HEIGHT_PX - y - config.ROW_HEIGHT, width * config.FIELD_WIDTH, config.ROW_HEIGHT,
              color='green')


def plot_vehicle(ax, x, y, width):
    plot_rect(ax, x, config.DISPLAY_HEIGHT_PX - y - config.ROW_HEIGHT, width * config.FIELD_WIDTH, config.ROW_HEIGHT,
              color='red')


def plot_player(ax, x, y):
    plot_rect(ax, x, config.DISPLAY_HEIGHT_PX - y - config.PLAYER_HEIGHT, config.PLAYER_WIDTH, config.PLAYER_HEIGHT,
              color='purple')


def draw_grid(ax):
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(config.FIELD_WIDTH))
    ax.yaxis.set_major_locator(MultipleLocator(config.ROW_HEIGHT))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#444444', linestyle='--')


def plot_state(ax, state, target_position, time=None):
    """ Plots a representation of the given state."""

    # plot background
    plot_target_position(ax, target_position)
    plot_water(ax)
    plot_street(ax)

    # plot entities
    for entry in reversed(state):  # reverse it so player will be drawn last and ON lilypads, not below them
        if entry[0] == 0:
            plot_player(ax, *entry[1:-1])
        elif entry[0] == 1:
            plot_vehicle(ax, *entry[1:])
        elif entry[0] == 2:
            plot_lilypad(ax, *entry[1:])
        else:
            raise RuntimeError(f"Unknown entity encoding in .npz-file: {entry[0]}")

    # plot properties
    if time:
        plt.title(f't={time}')
    plt.xlim((0, config.DISPLAY_WIDTH_PX))
    plt.ylim((0, config.DISPLAY_HEIGHT_PX))
    draw_grid(ax)