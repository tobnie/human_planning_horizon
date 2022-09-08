import os

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl

import config
from analysis import paper_plot_utils
from analysis.data_utils import get_street_data, get_river_data, subject2letter, read_data, transform_target_pos_to_string
from analysis.gaze.events.event_detection import try_fixation_detection
from analysis.gaze.vector_utils import calc_angle, calc_manhattan_distance, calc_euclidean_distance


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

    # fixation in field coordinates
    fix_info_df['fix_x_field'] = fix_info_df['fix_x'].floordiv(config.FIELD_WIDTH)
    fix_info_df['fix_y_field'] = fix_info_df['fix_y'].floordiv(config.FIELD_HEIGHT)

    fix_df.reset_index(inplace=True)
    fix_df.drop(['index', 'fix'], inplace=True, axis=1)
    fix_df = pd.concat([fix_df, fix_info_df], axis=1)

    fix_df.drop(['time_end'], axis=1)
    fix_df = fix_df[~(fix_df['fix_x'] == -32768)]
    return fix_df


def join_fixation_df_and_general_df(df, fixation_df):
    return df.merge(fixation_df, on=['subject_id', 'game_difficulty', 'world_number', 'time'], how='left').dropna(
        subset=['time_end', 'fix_duration', 'fix_x', 'fix_y'])


def add_fixation_distance_and_angle(df):
    df['fix_distance_euclidean'] = df.apply(lambda x: calc_euclidean_distance(x['player_x'], x['player_y'], x['fix_x'], x['fix_y']), axis=1)
    df['fix_distance_manhattan'] = df.apply(
        lambda x: calc_manhattan_distance(x['player_x_field'], x['player_y_field'], x['fix_x_field'], x['fix_y_field']), axis=1)
    df['fix_angle'] = df.apply(lambda x: calc_angle(x['player_x'], x['player_y'], x['fix_x'], x['fix_y']), axis=1)
    return df


def add_fixation_info_to_df(df):
    fixation_df = get_fixation_dataframe(df)
    df = join_fixation_df_and_general_df(df, fixation_df)
    df = add_fixation_distance_and_angle(df)

    # add weighted distance
    df = df.copy()
    df['fix_distance_euclidean_time'] = df['fix_distance_euclidean'].mul(df['fix_duration']).replace([np.inf, -np.inf], np.nan)
    df['fix_distance_manhattan_time'] = df['fix_distance_manhattan'].mul(df['fix_duration']).replace([np.inf, -np.inf], np.nan)

    summed_fixation_durations = df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
        'fix_duration'].agg(fix_duration_sum='sum').reset_index()

    # for euclidean distances
    summed_fixation_durations['fix_distance_euclidean_time_sum'] = \
        df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
            'fix_distance_euclidean_time'].agg(sum='sum').reset_index()['sum']

    # for manhattan distances
    summed_fixation_durations['fix_distance_manhattan_time_sum'] = \
        df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
            'fix_distance_manhattan_time'].agg(sum='sum').reset_index()['sum']

    # calculate final weighted value
    summed_fixation_durations['weighted_fix_distance_euclidean'] = summed_fixation_durations['fix_distance_euclidean_time_sum'].div(
        summed_fixation_durations['fix_duration_sum'])
    summed_fixation_durations['weighted_fix_distance_manhattan'] = summed_fixation_durations['fix_distance_manhattan_time_sum'].div(
        summed_fixation_durations['fix_duration_sum'])

    # for checking in debugger only:
    # df[(df['subject_id'] == 'AL09OL') & (df['game_difficulty'] == 'easy') & (df['world_number'] == 1) & (df['player_x_field'] == 8) & (df['player_y_field'] == 10)]

    # join summed fixations on original df
    df = df.merge(summed_fixation_durations, on=['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'],
                  how='left')
    df = df[
        ['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field', 'score', 'weighted_fix_distance_euclidean',
         'weighted_fix_distance_manhattan']].drop_duplicates()

    df = df[df['weighted_fix_distance_euclidean'].notna()]
    df.to_csv('fixations.csv')
    print('Saved Fixation Information')
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

    n = df.shape[0]
    df.loc[n, 'player_x_field'] = 5
    df.loc[n, 'player_y_field'] = 14
    df.loc[n, 'weighted_fix_distance_manhattan'] = 0.0

    weighted_fix_distance_pivot = pd.pivot_table(df, values='weighted_fix_distance_manhattan', index='player_y_field',
                                                 columns='player_x_field',
                                                 aggfunc=np.mean, fill_value=0, dropna=False)

    if subject_id:
        directory_path = './imgs/gaze/fixations/fixations_per_position/{}/'.format(subject_id)
    else:
        directory_path = './imgs/gaze/fixations/fixations_per_position/'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # plot heatmap distance
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.heatmap(weighted_fix_distance_pivot, ax=ax, vmax=3.5, cmap=paper_plot_utils.CMAP,
                cbar_kws={'label': 'Manhattan distance in fields'},
                linewidths=.1)
    ax.invert_yaxis()

    # if subject_id:
    #     plt.suptitle('Average Weighted Fixation Distance from Player - {}'.format(subject_id))
    # else:
    #     plt.suptitle('Average Weighted Fixation Distance from Player')

    plt.xlabel('')
    plt.ylabel('')

    ax.set_ylim((0, config.N_LANES))

    xlabels = [int(float(item.get_text()) + 1) for item in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)

    ylabels = [int(float(item.get_text()) + 1) for item in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)

    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_per_position.png')
    plt.savefig('../paper/fixation_distance_per_position.svg', format='svg')
    plt.show()


def plot_fixation_distance_per_position2(df, subject_id=None):
    if subject_id:
        df = df[df['subject_id'] == subject_id]

    if subject_id:
        directory_path = './imgs/gaze/fixations/fixations_per_position/{}/'.format(subject_id)
    else:
        directory_path = './imgs/gaze/fixations/fixations_per_position/'

    fix_angle_pivot = pd.pivot_table(df, values='weighted_fix_angle', index='player_y_field', columns='player_x_field',
                                     aggfunc=np.mean, fill_value=0)

    # ---plot per target
    g = sns.FacetGrid(df, col='target_position')
    g.map_dataframe(_plot_heatmap, 'weighted_fix_distance_manhattan', vmin=0, vmax=30)

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

    sns.catplot(data=df, x='experience', y='weighted_fix_distance_manhattan', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Experience')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_experience.png')

    sns.catplot(data=df, x='experience', y='weighted_fix_angle', kind='bar')
    plt.suptitle('Average Weighted Fixation Angle per Experience')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_experience.png')

    sns.catplot(data=df, x='score', y='weighted_fix_distance_manhattan', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_score.png')

    sns.catplot(data=df, x='score', y='weighted_fix_angle', kind='bar')
    plt.suptitle('Average Weighted Fixation Angle per Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_score.png')

    sns.catplot(data=df, x='subject_id', y='weighted_fix_distance_manhattan', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Subject')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_subject.png')

    sns.catplot(data=df, x='subject_id', y='weighted_fix_angle', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Subject')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_subject.png')


def get_region_from_field(y_field):
    if 1 <= y_field <= 7:
        return 'street'
    elif 9 <= y_field <= 13:
        return 'river'
    else:
        return np.nan


def plot_fixation_distance_hist_per_region(df):
    directory_path = './imgs/gaze/fixations/fixations_per_position/'

    df['region'] = df['player_y_field'].apply(get_region_from_field)

    fig, ax = plt.subplots()
    # multiple{“layer”, “dodge”, “stack”, “fill”}
    n_bins = 20
    x = df[df['region'] == 'street']['weighted_fix_distance_manhattan']
    ax.hist(x, density=True, bins=n_bins, label='street')
    y = df[df['region'] == 'river']['weighted_fix_distance_manhattan']
    ax.hist(y, density=True, bins=n_bins, alpha=0.5, label='river')
    ax.set_xscale('log')
    ax.legend()

    # df = df.drop(df[df['weighted_fix_distance_manhattan'] > 0.3].index)
    # binwidth = 0.04
    # sns.histplot(data=df, ax=ax, x="weighted_fix_distance_manhattan", hue="region", stat='proportion', multiple='fill', binwidth=binwidth, common_norm=False)  # multiple="stack",
    #
    # ax.set_xlabel('Average Fixation Distance [fields/ms]')
    # plt.savefig(directory_path + 'fixation_distance_per_region_hist.png')
    # plt.savefig('../paper/fixation_distance_per_region_hist.png')
    # plt.savefig('../paper/fixation_distance_per_region_hist.svg', format='svg')
    plt.show()


def plot_fixation_distance_box_per_region(df):
    directory_path = './imgs/gaze/fixations/fixations_per_position/'

    df['region'] = df['player_y_field'].apply(get_region_from_field)

    edge_colors = [paper_plot_utils.C0, paper_plot_utils.C1]
    box_colors = [paper_plot_utils.C0_soft, paper_plot_utils.C1_soft]

    # create boxplot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.boxplot(data=df, ax=ax, y="weighted_fix_distance_manhattan", x="region", width=0.2, linewidth=1.5,
                flierprops=dict(markersize=2),
                showmeans=True, meanline=True)

    # iterate over boxes
    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        patch.set_edgecolor(edge_colors[i])
        patch.set_facecolor(box_colors[i])

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(edge_colors[i])
            line.set_mfc(edge_colors[i])  # facecolor of fliers
            line.set_mec(edge_colors[i])  # edgecolor of fliers

    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('Fixation distance [fields/ms]')
    plt.savefig(directory_path + 'fixation_distance_per_region_box.png')
    plt.savefig('../paper/fixation_distance_per_region_box.svg', format="svg")
    plt.show()


def ttest_fixation_distance_street_river():
    df = pd.read_csv('fixations.csv')

    # get weighted fixation distances
    df_river = get_river_data(df)
    df_street = get_street_data(df)
    fix_distances_river = df_river['weighted_fix_distance_euclidean']
    fix_distances_street = df_street['weighted_fix_distance_euclidean']

    print(
        'H0: Mean fixations distances are equal | H1: Mean fixations distances on river are greater than mean fixation distances on street')

    # perform (Welch's) t-test
    # t test euclidean distances:
    ttest_result = scipy.stats.ttest_ind(fix_distances_river, fix_distances_street,
                                         alternative='greater')  # use equal_var=False bc of different sample sizes
    print('Test in Weighted Euclidean Distances')
    print(ttest_result)
    print('dof=', len(fix_distances_river) - 1 + len(fix_distances_street) - 1)

    fix_distances_river = df_river['weighted_fix_distance_manhattan']
    fix_distances_street = df_street['weighted_fix_distance_manhattan']

    # perform (Welch's) t-test
    # t test manhattan distances:
    ttest_result = scipy.stats.ttest_ind(fix_distances_river, fix_distances_street,
                                         alternative='greater')  # use equal_var=False bc of different sample sizes
    print('Test in Weighted Manhattan Distances')
    print(ttest_result)
    print('dof=', len(fix_distances_river) - 1 + len(fix_distances_street) - 1)


def kstest_fixation_distance_street_river():
    df = pd.read_csv('fixations.csv')

    # get weighted fixation distances
    df_river = get_river_data(df)
    df_street = get_street_data(df)
    fix_distances_river = df_river['weighted_fix_distance_euclidean']
    fix_distances_street = df_street['weighted_fix_distance_euclidean']

    print(
        'H0: Distributions for fixation distances are equal | H1: Distributions for fixation distances are different for river and street')

    # perform (Welch's) t-test
    # t test euclidean distances:
    kstest_result = scipy.stats.kstest(fix_distances_river, fix_distances_street, alternative='two-sided')
    print('Test in Weighted Euclidean Distances')
    print(kstest_result)

    fix_distances_river = df_river['weighted_fix_distance_manhattan']
    fix_distances_street = df_street['weighted_fix_distance_manhattan']

    # perform (Welch's) t-test
    # t test manhattan distances:
    kstest_result = scipy.stats.kstest(fix_distances_river, fix_distances_street, alternative='two-sided')
    print('Test in Weighted Manhattan Distances')
    print(kstest_result)


def plot_avg_fixation_distance_per_subject():
    df = pd.read_csv('fixations.csv')

    # TODO whats up with the scores per level
    level_scores_df = pd.read_csv('level_scores.csv').drop_duplicates()
    # order_df = level_scores_df.groupby(['subject_id'])['level_score'].mean().reset_index().sort_values('level_score')
    order_df = df[['subject_id', 'score']].drop_duplicates().sort_values('score')

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.pointplot(data=df, ax=ax, x='subject_id', y='weighted_fix_distance_manhattan', join=False, order=order_df['subject_id'])

    xlabels = [subject2letter(subj_id.get_text()) for subj_id in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = read_data()
    df = transform_target_pos_to_string(df)
    df = add_fixation_info_to_df(df)

    # df = pd.read_csv('fixations.csv')
    # plot_fixation_distance_per_position(df)
    # plot_fixation_distance_box_per_region(df)

    # # and for every subject:
    # for subject_id in get_all_subjects():
    #     print('For Subject ', subject_id)
    #     plot_fixation_distance_per_position(df, subject_id)
    #
    # plot_fixation_distance_per_meta_data(df)
    #
    # df_street = get_street_data(df)
    # plot_fixation_distance_per_meta_data(df_street, subfolder='street/')
    #
    # df_river = get_river_data(df)
    # plot_fixation_distance_per_meta_data(df_street, subfolder='river/')
