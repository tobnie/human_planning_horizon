from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


def plot_rect(ax, x, y, width, height, color):
    rect = Rectangle((x, y), width, height, color=color)
    ax.add_patch(rect)


def draw_grid(ax, multiple_x, multiple_y):
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(multiple_x))
    ax.yaxis.set_major_locator(MultipleLocator(multiple_y))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#444444', linestyle='--')
