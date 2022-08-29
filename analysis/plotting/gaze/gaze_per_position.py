import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from analysis.data_utils import read_data, read_subject_data, get_all_subjects, get_only_onscreen_data, position2field


def calculate_gaze_distance(gaze_x, gaze_y, player_x, player_y, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt((gaze_x - player_x) ** 2 + (gaze_y - player_y) ** 2)
    elif metric == 'manhattan':
        return np.abs(gaze_x - player_x) + np.abs(gaze_y - player_y)
    else:
        raise RuntimeError("Please specify a valid metric other than {}. Valid metrics are: \n 'euclidean', 'manhattan'".format(metric))


def calculate_gaze_distance_in_fields(gaze_x, gaze_y, player_x_field, player_y_field):
    gaze_x_field, gaze_y_field = position2field(gaze_x, gaze_y)
    return np.abs(gaze_x_field - player_x_field) + np.abs(gaze_y_field - player_y_field)


def calculate_gaze_angle_relative_to_player(gaze_x, gaze_y, player_x, player_y):
    return np.arctan2(gaze_y - player_y, gaze_x - player_x)


def calculate_avg_gaze_distance_per_field(df):
    df = df[['subject_id', 'gaze_x', 'gaze_y', 'player_x', 'player_y', 'player_x_field', 'player_y_field', 'target_position']].copy()

    # drop invalid samples
    df = df[df['gaze_x'] != -32768]

    # add gaze_distances to df
    df['gaze_distance'] = df[['gaze_x', 'gaze_y', 'player_x', 'player_y']].apply(lambda x: calculate_gaze_distance(*x), axis=1)
    df['gaze_distance_field'] = df[['gaze_x', 'gaze_y', 'player_x_field', 'player_y_field']].apply(
        lambda x: calculate_gaze_distance_in_fields(*x), axis=1)
    df['gaze_angle'] = df[['gaze_x', 'gaze_y', 'player_x', 'player_y']].apply(lambda x: calculate_gaze_angle_relative_to_player(*x), axis=1)

    # TODO also do for manhattan distance?: then we would also need to assign a field to the gaze
    # df['gaze_distance_manhattan'] = df[['gaze_x', 'gaze_y', 'player_x_field', 'player_y_field']].apply(lambda x: calculate_gaze_distance(*x, metric='manhattan'), axis=1)

    # group by field
    # TODO divide by sample count
    avg_gaze_per_field = df.groupby(['player_x_field', 'player_y_field'])['gaze_distance'].mean()
    avg_angle_per_field = df.groupby(['player_x_field', 'player_y_field'])['gaze_angle'].mean()

    return df


def plot_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    data_pivot = pd.pivot_table(data, values=args[0], index='player_y_field', columns='player_x_field', aggfunc=np.mean,
                                fill_value=0)

    if args[0] == 'gaze_angle':
        ax = sns.heatmap(data_pivot, vmin=-np.pi, vmax=np.pi, center=0),  # , annot=True, annot_kws={"fontsize": 8})
        # set color bar ticks
        # cbar = ax.collections[0].colorbar
        # cbar.set_ticks([-np.pi, 0, np.pi])
        # cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    else:
        ax = sns.heatmap(data_pivot)  # , annot=True, annot_kws={"fontsize": 8})

    plt.gca().invert_yaxis()


def plot_gaze_heatmap_per_position_of_player(df, subject_id=None):
    if subject_id:
        df = df[df['subject_id'] == subject_id]

    pos2str = {448: 'left', 1216: 'center', 2112: 'right'}
    df['target_position'] = df['target_position'].apply(lambda x: pos2str[x]).copy()

    print('min max distance: {} --- {}'.format(df['gaze_distance'].min(), df['gaze_distance'].max()))
    print('min max angle: {} --- {}'.format(df['gaze_angle'].min(), df['gaze_angle'].max()))

    gaze_pivot = pd.pivot_table(df, values='gaze_distance', index='player_y_field', columns='player_x_field', aggfunc=np.mean, fill_value=0)
    angle_pivot = pd.pivot_table(df, values='gaze_angle', index='player_y_field', columns='player_x_field', aggfunc=np.mean, fill_value=0)

    if subject_id:
        directory_path = './imgs/gaze/gaze_per_position/{}/'.format(subject_id)
    else:
        directory_path = './imgs/gaze/gaze_per_position/'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # plot heatmap distance
    sns.heatmap(gaze_pivot)  # , annot=True, annot_kws={"fontsize": 8})
    plt.gca().invert_yaxis()

    if subject_id:
        plt.savefig(directory_path + 'distance_per_position.png')
        plt.suptitle('Average Distance from Player to Gaze Sample - {}'.format(subject_id))
    else:
        plt.savefig('./imgs/gaze/gaze_per_position/distance_per_position.png')
        plt.suptitle('Average Distance from Player to Gaze Sample')
    plt.tight_layout()
    plt.show()

    # ---plot per target
    g = sns.FacetGrid(df, col='target_position')
    g.map_dataframe(plot_heatmap, 'gaze_distance')

    if subject_id:
        plt.suptitle('Average Distance from Player to Gaze Sample - {}'.format(subject_id))
        plt.savefig(directory_path + 'distance_per_position_by_target.png')
    else:
        plt.suptitle('Average Distance from Player to Gaze Sample')
        plt.savefig('./imgs/gaze/gaze_per_position/distance_per_position_by_target.png')

    plt.tight_layout()
    plt.show()

    # plot heatmap angle
    ax = sns.heatmap(angle_pivot, vmin=-np.pi, vmax=np.pi, center=0)  # , annot=True, annot_kws={"fontsize": 8})
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([-np.pi, 0, np.pi])
    # cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    plt.gca().invert_yaxis()
    plt.suptitle('Average Angle Of Gaze')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if subject_id:
        plt.savefig(directory_path + 'angle_per_position.png')
    else:
        plt.savefig('./imgs/gaze/gaze_per_position/angle_per_position.png')

    plt.show()

    # ---plot per target
    g = sns.FacetGrid(df, col='target_position')
    g.map_dataframe(plot_heatmap, 'gaze_angle')

    if subject_id:
        plt.savefig(directory_path + 'angle_per_position_by_target.png')
        plt.suptitle('Average Angle Of Gaze - {}'.format(subject_id))
    else:
        plt.savefig('./imgs/gaze/gaze_per_position/angle_per_position_by_target.png')
        plt.suptitle('Average Angle Of Gaze')

    plt.tight_layout()
    plt.show()


def plot_kde_with_player_position(*args, **kwargs):
    data = kwargs.pop('data')

    ax = sns.kdeplot(data=data, x='gaze_x', y='gaze_y', shade=True, cmap="viridis")

    # plot circle for player position
    # TODO also plot player position. The below code always plotted the same position
    # player_x = data['player_x_field'].iloc[0]
    # player_y = data['player_y_field'].iloc[0]


def plot_gaze_kde_per_player_position(df, subject_id=None):
    if subject_id:
        df = df[df['subject_id'] == subject_id]

    # only use gaze samples within screen:
    df = df[['gaze_x', 'gaze_y', 'player_x_field', 'player_y_field']].copy()
    df = get_only_onscreen_data(df).copy()
    df = df.reset_index()

    with sns.plotting_context('paper', font_scale=1.3):
        g = sns.FacetGrid(df, col="player_x_field", row="player_y_field", margin_titles=True,
                          row_order=reversed(range(int(df['player_y_field'].max()))))

    g.map_dataframe(plot_kde_with_player_position)
    g.set(xlim=(0, config.DISPLAY_WIDTH_PX), ylim=(0, config.DISPLAY_HEIGHT_PX))

    # plot player positions
    ax = g.axes
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            i_corrected = ax.shape[0] - i - 1
            player_x = j * config.FIELD_WIDTH + config.PLAYER_WIDTH / 2
            player_y = i_corrected * config.FIELD_HEIGHT + config.PLAYER_HEIGHT / 2
            player_circle = plt.Circle((player_x, player_y), 50, color='red')
            ax[i, j].add_patch(player_circle)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template='{row_name}', col_template='{col_name}')

    if subject_id:
        directory_path = './imgs/gaze/gaze_per_position/{}/'.format(subject_id)
    else:
        directory_path = './imgs/gaze/gaze_per_position/'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if subject_id:
        plt.suptitle('Gaze Density in the Level per Position - {}'.format(subject_id))
        plt.savefig(directory_path + 'gaze_density_per_position.png'.format(subject_id))
    else:
        plt.suptitle('Gaze Density in the Level per Position')
        plt.savefig('./imgs/gaze/gaze_per_position/gaze_density_per_position.png')
    plt.close(plt.gcf())


# TODO plot gaze distance (manhattan)

def run_gaze_per_position_plots():
    # TODO run for all subjects
    # print('Creating gaze per position plots for all data...')
    # df = read_data()
    # df_gaze_info = calculate_avg_gaze_distance_per_field(df)
    #
    # try:
    #     plot_gaze_heatmap_per_position_of_player(df_gaze_info)
    #     plot_gaze_kde_per_player_position(df)
    # except Exception as e:
    #     print('!!! -- Gaze Plots for all failed because of: \n' + str(e))

    df = read_data()
    df_gaze_info = calculate_avg_gaze_distance_per_field(df)
    print('Creating gaze per position plots for each subject separately...')
    for subject in tqdm(get_all_subjects()):
        try:
            plot_gaze_kde_per_player_position(df, subject_id=subject)
            plot_gaze_heatmap_per_position_of_player(df_gaze_info, subject_id=subject)
        except Exception as e:
            print('!!! -- Gaze Plots for {} failed because of: \n'.format(subject) + str(e))
