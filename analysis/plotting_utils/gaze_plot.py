import config


def plot_gaze(ax, times, coords):

    ax.plot(*coords, color='black')
    ax.scatter(*coords, color='black')

    ax.set_xlim(0, config.DISPLAY_WIDTH_PX)
    ax.set_ylim(0, config.DISPLAY_HEIGHT_PX)

