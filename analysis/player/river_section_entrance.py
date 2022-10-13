import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from analysis.data_utils import read_data
from analysis.gaze.fixations import get_region_from_field


def get_river_entrances_game(game_df):
    river_df = game_df[game_df['region'] == 'river']
    if river_df.empty:
        return None

    idx_diff = river_df.index.to_series().diff()

    river_entrances_idx = river_df.index[idx_diff > 1]
    river_first_entrance = river_df.head(1)

    river_entrances_x = river_df.loc[river_entrances_idx]
    all_river_entrances_x = pd.concat([river_entrances_x, river_first_entrance])
    return all_river_entrances_x


def get_river_entrances():
    """ Returns a dataframe containing all rows when the first lane of the river section was entered from below. """
    df = read_data()
    df['region'] = df['player_y_field'].apply(get_region_from_field)
    river_entrances = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(get_river_entrances_game).reset_index(drop=True)
    return river_entrances


def plot_entrance_of_river_section():
    entrances = get_river_entrances()
    x = entrances[['player_x_field', 'target_position']]
    ax = sns.histplot(data=x, x='player_x_field', hue='target_position', stat='proportion', multiple='stack', discrete=True)

    ax.set_xlabel('x')
    ax.set_ylabel('how often was river section entered at x')
    plt.legend(title='Target position', loc='upper right', labels=['left', 'center', 'right'])
    plt.savefig('./imgs/player_position/entrances_river_x_by_target_position.png')
    plt.savefig('../thesis/1descriptive/2position/river_entrances_OPT.png')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_entrance_of_river_section()
