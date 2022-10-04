import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.data_utils import read_data
from analysis.world.feature_maps import get_avoidance_map_from_state_identifier



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


def get_avoidance_map_distribution(group_df, size=7):
    if size not in [3, 5, 7]:
        raise NotImplementedError('Size must be 3, 5 or 7')

    identifiers = group_df['state_identifier']

    avoidance_maps = np.array(identifiers.apply(get_avoidance_map_from_state_identifier).tolist())
    avoidance_map_distribution = np.mean(avoidance_maps, axis=0)

    if size == 7:
        return avoidance_map_distribution  # TODO there seems to be an error in creation of feature maps, since sometimes player position is occupied
    else:
        # reshape and return then
        avoidance_map_distribution = avoidance_map_distribution.reshape((size, size))
        return avoidance_map_distribution


def heatmap_plot(*args, **kwargs):
    data = kwargs.pop('data')
    sns.heatmap(data['avoidance_distribution'].values[0])


def plot_actions_situation_heatmaps():
    situations = pd.read_csv('../data/situations.csv')

    # drop nan actions
    situations = situations.dropna(subset=['action'])
    avoidance_map_distributions = situations.groupby(['lane_type', 'action']).apply(
        lambda x: get_avoidance_map_distribution(x, 7)).reset_index(name='avoidance_distribution')

    # plot for each lane type and action
    g = sns.FacetGrid(avoidance_map_distributions, row='action', col='lane_type')
    g.map_dataframe(heatmap_plot)  # TODO fix feature maps
    plt.show()


if __name__ == '__main__':
    # plot_and_save_action_distributions()
    plot_actions_situation_heatmaps()
