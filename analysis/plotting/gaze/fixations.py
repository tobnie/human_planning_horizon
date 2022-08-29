import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import config
from analysis.data_utils import read_data, read_subject_data, position2field, get_all_subjects, transform_target_pos_to_string, \
    get_street_data, get_river_data
from analysis.plotting.gaze.vector_utils import calc_euclidean_distance, calc_angle, calc_manhattan_distance
from analysis.plotting.gaze.events.event_detection import try_fixation_detection


def position2field_row(row):
    row['fix_x_field'] = int(round(row['fix_x'] / config.FIELD_WIDTH))
    row['fix_y_field'] = int(round(row['fix_y'] / config.FIELD_HEIGHT))
    return row


def get_fixation_dataframe(df):
    experience_df = df[['subject_id', 'game_difficulty', 'world_number']].drop_duplicates()
    data = df.groupby(['subject_id', 'game_difficulty', 'world_number'])[['subject_id', 'experience', 'time', 'gaze_x', 'gaze_y']].agg(
        {'time': list, 'gaze_x': list, 'gaze_y': list}).reset_index()
    data = data.merge(experience_df, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    fix_df = data[['subject_id', 'game_difficulty', 'world_number', 'gaze_x', 'gaze_y', 'time']].copy()
    fix_df['fix'] = fix_df.apply(lambda x: try_fixation_detection(x['gaze_x'], x['gaze_y'], x['time'])[-1], axis=1)

    # remove rows where no saccades were detected
    fix_df['fix'] = fix_df['fix'].apply(lambda x: np.nan if len(x) == 0 else x)
    fix_df.dropna(subset=['fix'], inplace=True)

    # reformat df
    fix_df = fix_df.explode('fix')
    fix_df.drop(['gaze_x', 'gaze_y', 'time'], axis=1, inplace=True)

    fix_info_df = pd.DataFrame(fix_df['fix'].to_list(), columns=['time', 'time_end', 'fix_duration', 'fix_x', 'fix_y'])

    fix_df.reset_index(inplace=True)
    fix_df.drop(['index', 'fix'], inplace=True, axis=1)
    fix_df = pd.concat([fix_df, fix_info_df], axis=1)

    fix_df = fix_df.apply(position2field_row, axis=1)

    fix_df.drop(['time_end'], axis=1)
    fix_df = fix_df[~(fix_df['fix_x'] == -32768)]
    return fix_df


def join_fixation_df_and_general_df(df, fixation_df):
    return df.merge(fixation_df, on=['subject_id', 'game_difficulty', 'world_number', 'time'], how='left').dropna(
        subset=['time_end', 'fix_duration', 'fix_x', 'fix_y'])


def add_fixation_distance_and_angle(df):
    df['fix_distance'] = df.apply(lambda x: calc_euclidean_distance(x['player_x'], x['player_y'], x['fix_x'], x['fix_y']), axis=1)
    df['fix_distance_manhattan'] = df.apply(
        lambda x: calc_manhattan_distance(x['player_x_field'], x['player_y_field'], x['fix_x'], x['fix_y']), axis=1)
    df['fix_angle'] = df.apply(lambda x: calc_angle(x['player_x'], x['player_y'], x['fix_x'], x['fix_y']), axis=1)
    return df


def add_fixation_info_to_df(df):
    fixation_df = get_fixation_dataframe(df)
    df = join_fixation_df_and_general_df(df, fixation_df)
    df = add_fixation_distance_and_angle(df)

    # add weighted distance
    df = df.copy()
    df['weighted_fix_distance'] = df['fix_distance_manhattan'].div(df['fix_duration']).replace([np.inf, -np.inf], np.nan)
    df['weighted_fix_angle'] = df['fix_angle'].div(df['fix_duration']).replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=['weighted_fix_distance', 'weighted_fix_angle'], how='all', inplace=True)
    return df


def _plot_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    data_pivot = pd.pivot_table(data, values=args[0], index='player_y_field', columns='player_x_field', aggfunc=np.mean,
                                fill_value=0)

    if args[0] == 'fix_angle' or args[0] == 'weighted_fix_angle':
        ax = sns.heatmap(data_pivot, center=0)
    else:
        ax = sns.heatmap(data_pivot)  # , annot=True, annot_kws={"fontsize": 8})

    ax.invert_yaxis()


def plot_fixation_distance_per_position(df, subject_id=None):
    if subject_id:
        df = df[df['subject_id'] == subject_id]

    weighted_fix_distance_pivot = pd.pivot_table(df, values='weighted_fix_distance', index='player_y_field', columns='player_x_field',
                                                 aggfunc=np.mean, fill_value=0)
    fix_angle_pivot = pd.pivot_table(df, values='weighted_fix_angle', index='player_y_field', columns='player_x_field',
                                     aggfunc=np.mean, fill_value=0)

    if subject_id:
        directory_path = './imgs/gaze/fixations/fixations_per_position/{}/'.format(subject_id)
    else:
        directory_path = './imgs/gaze/fixations/fixations_per_position/'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # plot heatmap distance
    sns.heatmap(weighted_fix_distance_pivot)  # , annot=True, annot_kws={"fontsize": 8})
    plt.gca().invert_yaxis()

    if subject_id:
        plt.suptitle('Average Weighted Fixation Distance from Player - {}'.format(subject_id))
    else:
        plt.suptitle('Average Weighted Fixation Distance from Player')

    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_per_position.png')
    plt.show()

    # ---plot per target
    g = sns.FacetGrid(df, col='target_position')
    g.map_dataframe(_plot_heatmap, 'weighted_fix_distance', vmin=0, vmax=30)

    if subject_id:
        plt.suptitle('Average Weighted Fixation Distance from Player - {}'.format(subject_id))
    else:
        plt.suptitle('Average Weighted Fixation Distance from Player')

    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_per_position_by_target.png')
    plt.show()

    # plot heatmap angle
    ax = sns.heatmap(fix_angle_pivot, center=0)  # , annot=True, annot_kws={"fontsize": 8})
    ax.invert_yaxis()

    if subject_id:
        plt.suptitle('Average Weighted Angle from Fixation to Player - {}'.format(subject_id))
    else:
        plt.suptitle('Average Weighted Angle from Fixation to Player')

    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_per_position.png')
    plt.show()

    # ---plot per target
    g = sns.FacetGrid(df, col='target_position')
    g.map_dataframe(_plot_heatmap, 'weighted_fix_angle')

    if subject_id:
        plt.suptitle('Average Weighted Angle from Fixation to Player - {}'.format(subject_id))
    else:
        plt.suptitle('Average Weighted Angle from Fixation to Player')

    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_per_position_by_target.png')
    plt.show()


def plot_fixation_distance_per_meta_data(df, subfolder=''):
    directory_path = './imgs/gaze/fixations/fixations_per_position/' + subfolder

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    sns.catplot(data=df, x='experience', y='weighted_fix_distance', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Experience')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_experience.png')

    sns.catplot(data=df, x='experience', y='weighted_fix_angle',  kind='bar')
    plt.suptitle('Average Weighted Fixation Angle per Experience')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_experience.png')

    sns.catplot(data=df, x='score', y='weighted_fix_distance',  kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_score.png')

    sns.catplot(data=df, x='score', y='weighted_fix_angle',  kind='bar')
    plt.suptitle('Average Weighted Fixation Angle per Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_score.png')

    sns.catplot(data=df, x='subject_id', y='weighted_fix_distance',  kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Subject')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_subject.png')

    sns.catplot(data=df, x='subject_id', y='weighted_fix_angle',  kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Subject')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_subject.png')


df = read_data()
df = transform_target_pos_to_string(df)
df = add_fixation_info_to_df(df)
# plot_fixation_distance_per_position(df)
#
# # and for every subject:
# for subject_id in get_all_subjects():
#     print('For Subject ', subject_id)
#     plot_fixation_distance_per_position(df, subject_id)

plot_fixation_distance_per_meta_data(df)

df_street = get_street_data(df)
plot_fixation_distance_per_meta_data(df_street, subfolder='street/')

df_river = get_river_data(df)
plot_fixation_distance_per_meta_data(df_street, subfolder='river/')
