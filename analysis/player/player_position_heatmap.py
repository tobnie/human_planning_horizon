import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.data_utils import coords2fieldsx, coords2fieldsy, read_data


def add_player_position_in_field_coordinates(df):
    df['player_x_field'] = df['player_x'].apply(coords2fieldsx)
    df['player_y_field'] = df['player_y'].apply(coords2fieldsy)
    return df


def plot_player_position_heatmap(df=None):
    if df is None:
        df = read_data()

    position_df = df[['subject_id', 'player_x_field', 'player_y_field']].copy()

    # TODO
    # # add last row empty
    # n = df.shape[0]
    # df.loc[n, 'player_x_field'] = 5
    # df.loc[n, 'player_y_field'] = 14

    ax = heatmap_df = pd.crosstab(position_df['player_y_field'], position_df['player_x_field'])
    sns.heatmap(heatmap_df)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.title('player position heatmap')
    plt.savefig('./imgs/player_position/player_position_heatmap.png')
    plt.close(plt.gcf())

    # again with player start position set to zero
    heatmap_df.iloc[0, 9] = 0
    ax = sns.heatmap(heatmap_df)
    ax.invert_yaxis()
    plt.title('start location set to zero for scaling of color')
    plt.tight_layout()
    plt.savefig('./imgs/player_position/player_position_heatmap_start_position_set_to_zero.png')
    plt.close(plt.gcf())

    # TODO also only with won games so mass is not distributed over first few rows primarily
    df_won_games = df.loc[df['game_status'] == 'won']
    position_df_won = df_won_games[['player_x_field', 'player_y_field']].copy()
    position_df_won.dropna(inplace=True)  # TODO when does NaN occur?

    heatmap_df = pd.crosstab(position_df_won['player_y_field'], position_df_won['player_x_field'])
    heatmap_df.iloc[0, 9] = 0  # set start position to zero
    ax = sns.heatmap(heatmap_df)
    ax.invert_yaxis()
    plt.title('only won games')
    plt.tight_layout()
    plt.savefig('./imgs/player_position/player_position_heatmap_start_position_set_to_zero_only_won_games.png')
    plt.close(plt.gcf())


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
    df = read_data()
    plot_player_position_heatmap(df)
    plot_position_heatmap_per_player(df)
