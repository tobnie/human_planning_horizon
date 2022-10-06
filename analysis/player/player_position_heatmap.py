import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config
from analysis import paper_plot_utils
from analysis.data_utils import coords2fieldsx, coords2fieldsy, read_data


def add_player_position_in_field_coordinates(df):
    df['player_x_field'] = df['player_x'].apply(coords2fieldsx)
    df['player_y_field'] = df['player_y'].apply(coords2fieldsy)
    return df


def plot_player_position_heatmap(df=None):
    if df is None:
        df = read_data()

    position_df = df[['subject_id', 'player_x_field', 'player_y_field']].copy()

    # TODO add last row empty
    # n = df.shape[0]
    # df.loc[n, 'player_x_field'] = 5
    # df.loc[n, 'player_y_field'] = 14

    heatmap_df = pd.crosstab(position_df['player_y_field'], position_df['player_x_field'], normalize=True)
    ax = sns.heatmap(heatmap_df, cbar_kws={'label': '% time on each field'}, linewidths=.1)
    ax.invert_yaxis()

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('./imgs/player_position/player_position_heatmap.png')
    plt.close(plt.gcf())

    # again with player start 2position set to zero
    heatmap_df.iloc[0, 9] = 0
    ax = sns.heatmap(heatmap_df, cbar_kws={'label': '% time on each field'}, linewidths=.1)
    ax.invert_yaxis()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('start location set to zero for scaling of color')
    plt.tight_layout()
    plt.savefig('./imgs/player_position/player_position_heatmap_start_position_set_to_zero.png')
    plt.close(plt.gcf())

    df_won_games = df.loc[df['game_status'] == 'won']
    position_df_won = df_won_games[['player_x_field', 'player_y_field']].copy()
    position_df_won.dropna(inplace=True)

    heatmap_df = pd.crosstab(position_df_won['player_y_field'], position_df_won['player_x_field'])
    heatmap_df.iloc[0, 9] = 0  # set start position to zero
    ax = sns.heatmap(heatmap_df, cbar_kws={'label': '% time on each field'}, linewidths=.1)
    ax.invert_yaxis()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig('./imgs/player_position/player_position_heatmap_start_position_set_to_zero_only_won_games.png')
    plt.savefig('../thesis/1descriptive/2position/player_position_heatmap.png')
    plt.close(plt.gcf())


def plot_player_position_heatmap_per_target_position():
    df = read_data()

    position_df = df[['player_x_field', 'player_y_field', 'target_position']].copy()

    position_df['target_position'] = position_df['target_position'].replace({448: 'left', 1216: 'center', 2112: 'right'})

    # TODO add last row empty
    # n = df.shape[0]
    # df.loc[n, 'player_x_field'] = 5
    # df.loc[n, 'player_y_field'] = 14

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])  # TODO right subplot is smaller than first two
    for i, target_pos in enumerate(position_df['target_position'].unique()):
        ax = axs[i]
        target_position_df = position_df[position_df['target_position'] == target_pos]
        target_position_df.drop('target_position', axis='columns', inplace=True)
        heatmap_df = pd.crosstab(target_position_df['player_y_field'], target_position_df['player_x_field'], normalize=True)

        heatmap_df.iloc[0, 9] = 0  # set start position to zero
        if i == 2:
            sns.heatmap(heatmap_df, vmax=0.025, cbar_kws={'label': '% time on each field'}, cbar=cbar_ax, linewidths=.1, ax=ax)
        else:
            sns.heatmap(heatmap_df, vmax=0.025, cbar=None, linewidths=.1, ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel('x')

        if i == 0:
            ax.set_ylabel('y')
            ax.set_yticks(range(0, config.N_LANES))
        else:
            ax.set_ylabel('')
            ax.set_yticks([])
        ax.set_title(f'target={target_pos}')

    fig.axes[-2].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'../thesis/1descriptive/2position/player_position_heatmap_by_target_pos.png')
    plt.show()


def _plot_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    data_pivot = pd.crosstab(data['player_y_field'], data['player_x_field'])

    data_pivot.iloc[0, 9] = 0
    ax = sns.heatmap(data_pivot)  # , annot=True, annot_kws={"fontsize": 8})
    ax.invert_yaxis()


def plot_position_heatmap_per_player(df=None):
    if df is None:
        df = read_data()

    position_df = df[['subject_id', 'player_x_field', 'player_y_field']].copy()

    g = sns.FacetGrid(position_df, col='subject_id', col_wrap=4)
    g.map_dataframe(_plot_heatmap)

    plt.tight_layout()
    plt.savefig('./imgs/player_position/player_position_heatmap_per_player.png')
    plt.show()


if __name__ == '__main__':
    plot_player_position_heatmap()
    plot_player_position_heatmap_per_target_position()
    # plot_position_heatmap_per_player()
