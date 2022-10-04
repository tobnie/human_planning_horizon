from matplotlib import pyplot as plt

import config
from analysis import plotting_utils
from analysis.plotting_utils import plot_rect


def plot_target_position(ax, target_position):
    plot_rect(ax, target_position * config.FIELD_WIDTH, (config.N_LANES - 1) * config.ROW_HEIGHT, config.FIELD_WIDTH,
              config.ROW_HEIGHT, 'pink')


def plot_water(ax):
    water_width = config.DISPLAY_WIDTH_PX
    water_height = config.N_WATER_LANES * config.ROW_HEIGHT
    water_y = (config.N_STREET_LANES + 2) * config.ROW_HEIGHT
    plot_rect(ax, 0, water_y, water_width, water_height, color='lightblue')


def plot_street(ax):
    street_width = config.DISPLAY_WIDTH_PX
    street_height = config.N_STREET_LANES * config.ROW_HEIGHT
    street_y = config.ROW_HEIGHT
    plot_rect(ax, 0, street_y, street_width, street_height, color='gray')


def plot_lilypad(ax, x, y, width):
    plot_rect(ax, x, y, width, config.ROW_HEIGHT,
              color='green')


def plot_vehicle(ax, x, y, width):
    plot_rect(ax, x, y, width, config.ROW_HEIGHT,
              color='red')


def plot_player(ax, x, y):
    plot_rect(ax, x, y, config.PLAYER_WIDTH, config.PLAYER_HEIGHT,
              color='purple')


def plot_world_background(ax, target_position=None):
    if target_position:
        plot_target_position(ax, target_position)
    plot_water(ax)
    plot_street(ax)


def draw_grid(ax):
    plotting_utils.draw_grid(ax, config.FIELD_WIDTH, config.ROW_HEIGHT)


def plot_state(ax, state, target_position, time=None):
    """ Plots a representation of the given state."""

    # plot background
    plot_world_background(ax, target_position)

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


def plot_line_between_points(ax, x1, y1, x2, y2, color='black'):
    ax.plot([x1, x2], [y1, y2], color=color)


def get_center_of_entity(x, y, width, height):
    return x + width / 2, y + height / 2


def plot_player_path(ax, states, target_position):
    # plot background
    plot_world_background(ax, target_position)

    player_pos = states[0][1][0][1:]
    player_pos = player_pos[0], config.DISPLAY_HEIGHT_PX - player_pos[1] - config.PLAYER_HEIGHT
    player_pos = get_center_of_player(*player_pos)

    for time_state in states[1:]:
        time = time_state[0]
        state = time_state[1]
        player_state = state[0]
        player_pos_next = player_state[1:]
        player_pos_next = player_pos_next[0], config.DISPLAY_HEIGHT_PX - player_pos_next[1] - config.PLAYER_HEIGHT
        player_pos_next = get_center_of_player(*player_pos_next)

        if player_pos[0] != player_pos_next[0] or player_pos[1] != player_pos_next[1]:
            # print(f'Line from {player_pos} to {player_pos_next}')
            plot_line_between_points(ax, player_pos[0], player_pos[1], player_pos_next[0], player_pos_next[1], color='k')

        player_pos = player_pos_next

    plt.title('Player Path')
    plt.xlim((0, config.DISPLAY_WIDTH_PX))
    plt.ylim((0, config.DISPLAY_HEIGHT_PX))
    draw_grid(ax)


def get_center_of_player(x, y):
    return get_center_of_entity(x, y, config.PLAYER_WIDTH, config.PLAYER_HEIGHT)
