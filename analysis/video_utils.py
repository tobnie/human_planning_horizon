import os

from matplotlib import pyplot as plt
from tqdm import tqdm

from analysis.world.world_coordinates import plot_state


def generate_plots(time_states, target_position, path=f'./test/'):
    """ Generates world state plots for every state in states and saves them
    individually at the given path location. """
    print('Generating frames for animation of gameplay video...')
    for i, (time, state) in enumerate(tqdm(time_states)):
        fig, ax = plt.subplots()
        plot_state(ax, state, target_position)
        plt.savefig(path + f'{i}.png')
        plt.close()


def animate_plots(path):
    imgs = [plt.imread(path + img) for img in os.listdir(path) if img.endswith('.png')]
    fig = plt.figure()
    viewer = fig.add_subplot(111)
    plt.ion()  # Turns interactive mode on (probably unnecessary)

    for i in range(len(imgs)):
        viewer.clear()  # Clears the previous image
        viewer.imshow(imgs[i])  # Loads the new image
        plt.pause(.1)  # Delay in seconds
        fig.canvas.draw()  # Draws the image to the screen
    plt.close()
