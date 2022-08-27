import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.data_utils import read_data, drop_missing_samples


def get_diff_from_gaze_to_player(df):
    df = df[['gaze_x', 'gaze_y', 'player_x', 'player_y']]
    gaze_x, gaze_y = df['gaze_x'], df['gaze_y']
    player_x, player_y = df['player_x'], df['player_y']

    diff_x = gaze_x - player_x
    diff_y = gaze_y - player_y
    diff_total = np.sqrt(diff_x ** 2 + diff_y ** 2)

    # Create new pandas DataFrame.
    df = df.assign(dx=diff_x, dy=diff_y, d_total=diff_total)

    return df


def plot_diff_from_gaze_to_player(df):
    df_plot = df[['player_y', 'dx', 'dy', 'd_total']]
    df_plot = pd.melt(df_plot, id_vars="player_y", var_name="axis", value_name="delta")
    sns.catplot(x='player_y', y='delta', hue='axis', data=df_plot, kind='bar')
    plt.show()


def scatter_plot_diff_from_gaze_to_player(df):
    sns.scatterplot(x='dx', y='dy', data=df)    # TODO style / markers for smaller scatter point size?
    plt.show()


df = read_data()
print(df['player_y'].unique())
df = drop_missing_samples(df)
df = get_diff_from_gaze_to_player(df)
plot_diff_from_gaze_to_player(df)
scatter_plot_diff_from_gaze_to_player(df)
