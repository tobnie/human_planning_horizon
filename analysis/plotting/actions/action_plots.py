import numpy as np
from matplotlib import pyplot as plt

from analysis.analysis_utils import get_times_actions, get_all_times_actions_of_player, get_all_subjects
from game.world.player import PlayerAction
from game.world_generation.generation_config import GameDifficulty


def action_int_to_name(action_int):
    return PlayerAction(action_int).name.lower()


def plot_action_distribution(ax, actions, title='Distribution of actions'):
    action_counts = np.unique(actions, return_counts=True)
    all_actions = list(range(len(PlayerAction)))
    all_action_counts = [0] * len(all_actions)

    for a, c in zip(*action_counts):
        all_action_counts[int(a)] = c

    ax.bar(all_actions, all_action_counts)

    ax.set_xticks(range(len(PlayerAction)))
    ax.set_xticklabels(labels=[action_int_to_name(i) for i, tick in enumerate(ax.get_xticklabels())])
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title(title)

test = get_all_subjects()
for subject in get_all_subjects():
    times_actions = get_all_times_actions_of_player(subject)
    times, actions = list(zip(*times_actions))

    fig, ax = plt.subplots()
    plot_action_distribution(ax, actions, title='Distribution of actions for {}'.format(subject))
    plt.savefig('./imgs/{}_action_distribution.png'.format(subject))
