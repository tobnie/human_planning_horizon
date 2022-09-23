import numpy as np
from matplotlib import pyplot as plt

from analysis.data_utils import read_data
from game.world.player import PlayerAction


def action_int_to_name(action_int):
    return PlayerAction(action_int).name.lower()


def plot_action_distribution(ax, actions, title='Distribution of actions'):
    action_counts = np.unique(actions, return_counts=True)
    ax.bar(action_counts[0], action_counts[1])
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title(title)


def plot_and_save_action_distributions():
    df = read_data()
    for subject in df.subject_id.unique():
        fig, ax = plt.subplots()
        df_subject = df[df.subject_id == subject]
        actions = df_subject.loc[:, 'action'].to_numpy()
        plot_action_distribution(ax, actions, title='Distribution of actions for {}'.format(subject))
        plt.savefig('./imgs/actions/{}_action_distribution.png'.format(subject))
        plt.close(fig)


plot_and_save_action_distributions()
