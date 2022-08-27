import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config
from analysis.data_utils import read_data
from analysis.plotting.performance.performances import add_game_status_to_df


def coords2fieldsx(x):
    return min(config.N_FIELDS_PER_LANE - 1, max(x // config.FIELD_WIDTH, 0))


def coords2fieldsy(y):
    return y // config.FIELD_HEIGHT


def add_player_position_in_field_coordinates(df):
    df['player_x_field'] = df['player_x'].apply(coords2fieldsx)
    df['player_y_field'] = df['player_y'].apply(coords2fieldsy)
    return df


def plot_player_position_heatmap(df=None):
    if df is None:
        df = read_data()

    df = add_game_status_to_df(df)
    df = add_player_position_in_field_coordinates(df)
    df['player_x'] = df['player_x'].apply(coords2fieldsx)
    df['player_y'] = df['player_y'].apply(coords2fieldsy)
    position_df = df[['player_x', 'player_y']].copy()
    position_df.dropna(inplace=True)  # TODO when does NaN occur?

    # TODO player_x = -1 ???
    heatmap_df = pd.crosstab(position_df['player_y'], position_df['player_x'])
    sns.heatmap(heatmap_df)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.title('heatmap')
    plt.savefig('./player_position/player_position_heatmap.png')
    plt.close(plt.gcf())

    # again with player start position set to zero
    heatmap_df.iloc[0, 10] = 0
    sns.heatmap(heatmap_df)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.title('start location set to zero for scaling of color')
    plt.savefig('./player_position/player_position_heatmap_start_position_set_to_zero_only_won_games.png')
    plt.close(plt.gcf())

    # TODO also only with won games so mass is not distributed over first few rows primarily
    df_won_games = df.loc[df['game_status'] == 'won']
    position_df_won = df_won_games[['player_x', 'player_y']].copy()
    position_df_won.dropna(inplace=True)  # TODO when does NaN occur?

    heatmap_df = pd.crosstab(position_df_won['player_y'], position_df_won['player_x'])
    heatmap_df.iloc[0, 10] = 0  # set start position to zero
    sns.heatmap(heatmap_df)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.title('only won games')
    plt.savefig('./player_position/player_position_heatmap_start_position_set_to_zero_only_won_games.png')
    plt.close(plt.gcf())


def plot_position_heatmap_per_player(df=None):
    if df is None:
        df = read_data()

    position_df = df[['subject_id', 'player_x', 'player_y']].copy()
    # sort the dataframe
    position_df.sort_values(by='subject_id', axis=1, inplace=True)

    # set the index to be this and don't drop
    position_df.set_index(keys=['subject_id'], drop=False, inplace=True)

    # get a list of names
    subject_ids = position_df['subject_id'].unique().tolist()

    fig, ax = plt.subplots(len(subject_ids))
    for i, id in enumerate(subject_ids):
        # now we can perform a lookup on a 'view' of the dataframe
        subject_df = position_df.loc[position_df.name == id]
        sns.heatmap(subject_df[['player_x', 'player_y']], ax=ax[i])
        ax.set_title(id)
    plt.show()

#
# df = read_data()
# plot_player_position_heatmap(df)
# plot_position_heatmap_per_player(df)
