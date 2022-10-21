import os
from itertools import product

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt, patches
import seaborn as sns
import matplotlib as mpl

import config
from analysis import paper_plot_utils
from analysis.data_utils import coords2fieldsx, get_street_data, get_river_data, subject2letter, read_data
from analysis.gaze.events.event_detection import try_fixation_detection
from analysis.gaze.vector_utils import calc_manhattan_distance, calc_euclidean_distance, calc_angle_relative_to_front


def load_fixations(remodnav=True):
    # load data
    if remodnav:
        df = pd.read_csv('../data/fixations_remodnav.csv')
    else:
        df = pd.read_csv('../data/fixations.csv')

    return df


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
    df['fix_angle'] = df.apply(
        lambda x: calc_angle_relative_to_front(x['player_x_field'], x['player_y_field'], x['fix_x_field'], x['fix_y_field']), axis=1)
    return df


def attribute_fixation(row, radius):
    """ Attributes a fixation to the player or target if the fixation is on player or target or near_{player/target} if the fixation is
    within the radius of player or target, otherwise to the world.
    If the fixation is within the vicinity of the player and the target, attribute the fixation to the nearer object in px by the
    euclidean norm."""

    # get player 2position
    p_x = int(row['player_x_field'])
    p_y = int(row['player_y_field'])
    p_range_x = range(p_x - radius, p_x + radius + 1)
    p_range_y = range(p_y - radius, p_y + radius + 1)

    # get target 2position
    target_x = row['target_position']

    target_field_x = int(coords2fieldsx(target_x))
    target_field_y = config.N_LANES - 1
    target_range_x = range(target_field_x - radius, target_field_x + radius + 1)
    target_range_y = range(target_field_y - radius, config.N_LANES - 1)

    # get fixation
    f_x = int(row['fix_x_field'])
    f_y = int(row['fix_y_field'])

    # check if in range
    if f_x == p_x and f_y == p_y:
        row['fixation_on'] = 'player'
    elif f_x == target_field_x and f_y == target_field_y:
        row['fixation_on'] = 'target'
    elif f_x in p_range_x and f_y in p_range_y and f_x in target_range_x and f_y in target_range_y:
        # if it is in range of target and player, attribute to nearer object in pixels instead of fields
        fix_pos = np.array([row['fix_x'], row['fix_y']])
        player_pos = np.array([row['player_x'], row['player_y']])
        target_y = config.DISPLAY_HEIGHT_PX - config.FIELD_HEIGHT / 2
        target_pos = np.array([target_x, target_y])
        dist_to_player = np.linalg.norm(fix_pos - player_pos)
        dist_to_target = np.linalg.norm(fix_pos - target_pos)
        if dist_to_player < dist_to_target:
            row['fixation_on'] = 'near_player'
        else:
            row['fixation_on'] = 'near_target'
    elif f_x in p_range_x and f_y in p_range_y:
        row['fixation_on'] = 'near_player'
    elif f_x in target_range_x and f_y in target_range_y:
        row['fixation_on'] = 'near_target'
    else:
        row['fixation_on'] = 'world'

    return row


def attribute_fixations_in_df(df, radius=1):
    """ Attributes the fixation to the player, the target 2position or the game objects by considering a radius around the fixated field."""
    df = df.apply(lambda x: attribute_fixation(x, radius), axis=1)
    return df


def save_fixations():
    df = read_data()
    df = add_fixation_info_to_df(df)
    df = attribute_fixations_in_df(df)
    df.to_csv('fixations.csv', index=False)
    print('Saved Fixation Information')
    return df


def add_fixation_info_to_df(df, drop_offscreen_samples=True):
    fixation_df = get_fixation_dataframe(df)
    df = join_fixation_df_and_general_df(df, fixation_df)
    df = add_fixation_distance_and_angle(df)

    # add weighted distance
    df = df.copy()
    # df['fix_distance_euclidean_time'] = df['fix_distance_euclidean'].mul(df['fix_duration']).replace([np.inf, -np.inf], np.nan)
    df['fix_distance_manhattan_time'] = df['fix_distance_manhattan'].mul(df['fix_duration']).replace([np.inf, -np.inf], np.nan)
    df['fix_angle_time'] = df['fix_angle'].mul(df['fix_duration']).replace([np.inf, -np.inf], np.nan)

    summed_fixation_durations = df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
        'fix_duration'].agg(fix_duration_sum='sum').reset_index()

    # for euclidean distances
    # summed_fixation_durations['fix_distance_euclidean_time_sum'] = \
    #     df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
    #         'fix_distance_euclidean_time'].agg(sum='sum').reset_index()['sum']

    # for manhattan distances
    summed_fixation_durations['fix_distance_manhattan_time_sum'] = \
        df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
            'fix_distance_manhattan_time'].agg(sum='sum').reset_index()['sum']

    # for angle
    summed_fixation_durations['fix_angle_time_sum'] = \
        df.groupby(['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'])[
            'fix_angle_time'].agg(sum='sum').reset_index()['sum']

    # calculate final weighted value
    # summed_fixation_durations['weighted_fix_distance_euclidean'] = summed_fixation_durations['fix_distance_euclidean_time_sum'].div(
    #     summed_fixation_durations['fix_duration_sum'])
    summed_fixation_durations['mfd'] = summed_fixation_durations['fix_distance_manhattan_time_sum'].div(
        summed_fixation_durations['fix_duration_sum'])
    summed_fixation_durations['mfa'] = summed_fixation_durations['fix_angle_time_sum'].div(
        summed_fixation_durations['fix_duration_sum'])

    # join summed fixations on original df
    df = df.merge(summed_fixation_durations, on=['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field'],
                  how='left')
    df = df[
        ['subject_id', 'game_difficulty', 'world_number', 'time', 'target_position', 'player_x', 'player_y', 'player_x_field',
         'player_y_field', 'region', 'score',
         'mfd', 'mfa', 'state', 'fix_x', 'fix_y', 'fix_x_field', 'fix_y_field',
         'fix_distance_manhattan', 'fix_angle', 'fix_duration']].drop_duplicates()

    df = df[df['mfd'].notna()]

    # drop fixations not within the screen limits
    if drop_offscreen_samples:
        mask = (df['fix_x'] <= config.DISPLAY_WIDTH_PX) & (df['fix_x'] >= 0) & (df['fix_y'] >= 0) & (
                df['fix_y'] <= config.DISPLAY_HEIGHT_PX)
        df = df[mask]

    return df


def _plot_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    data_pivot = pd.pivot_table(data, values=args[0], index='player_y_field', columns='player_x_field', aggfunc=np.mean,
                                fill_value=0)

    if args[0] == 'fix_angle' or args[0] == 'mfa':
        ax = sns.heatmap(data_pivot, center=0)
    else:
        ax = sns.heatmap(data_pivot)  # , annot=True, annot_kws={"fontsize": 8})

    ax.invert_yaxis()


def plot_mfd_heatmap(subject_id=None):
    df = load_fixations()

    if subject_id:
        df = df[df['subject_id'] == subject_id]

    n = df.shape[0]
    df.loc[n, 'player_x_field'] = 5
    df.loc[n, 'player_y_field'] = 14
    df.loc[n, 'mfd'] = 0.0

    weighted_fix_distance_pivot = pd.pivot_table(df, values='mfd', index='player_y_field',
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

    plt.xlabel('')
    plt.ylabel('')

    ax.set_ylim((0, config.N_LANES))

    xlabels = [int(float(item.get_text()) + 1) for item in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)

    ylabels = [int(float(item.get_text()) + 1) for item in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)

    plt.tight_layout()
    plt.savefig(directory_path + 'mfd_per_position.png')
    plt.savefig('../thesis/2river_vs_street/1mfd/mfd_per_position.png')
    plt.savefig('../paper/mfd_per_position.svg', format='svg')
    plt.show()


def plot_fixation_heatmap():
    fixations = load_fixations()
    df = read_data()[['subject_id', 'game_difficulty', 'world_number', 'game_status']].drop_duplicates()

    fixations = fixations.merge(df, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    fixations = fixations[fixations['game_status'] == 'won']

    fixations = fixations[['fix_x_field', 'fix_y_field', 'fix_duration']]
    fix_duration_sum = fixations['fix_duration'].sum()
    fixations = fixations.groupby(['fix_x_field', 'fix_y_field'])['fix_duration'].sum().reset_index()
    fixations_pivot = fixations.pivot(index='fix_y_field', columns='fix_x_field', values='fix_duration')

    # normalize pivot table
    fixations_pivot = fixations_pivot.div(fix_duration_sum).fillna(0)

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.heatmap(fixations_pivot, ax=ax, cbar_kws={'label': '% fixation duration of all fixations'}, linewidths=.1)
    ax.invert_yaxis()

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()
    plt.savefig('../thesis/1descriptive/3gaze/fixations_heatmap.png')
    plt.show()


def plot_fixation_angle_per_position(df, subject_id=None):
    if subject_id:
        df = df[df['subject_id'] == subject_id]

    n = df.shape[0]
    df.loc[n, 'player_x_field'] = 5
    df.loc[n, 'player_y_field'] = 14
    df.loc[n, 'mfa'] = 0.0

    # TODO fill value with nan and replace with mean value?
    mfa_pivot = pd.pivot_table(df, values='mfa', index='player_y_field',
                               columns='player_x_field',
                               aggfunc=np.mean, fill_value=0, dropna=False)

    if subject_id:
        directory_path = './imgs/gaze/fixations/fixations_per_position/{}/'.format(subject_id)
    else:
        directory_path = './imgs/gaze/fixations/fixations_per_position/'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # plot heatmap distance
    fig, ax = plt.subplots(figsize=(12, 12))  # paper_plot_utils.figsize)
    sns.heatmap(mfa_pivot, ax=ax, cmap=paper_plot_utils.CMAP, annot=True,
                cbar_kws={'label': 'Angle'}, linewidths=.1)
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
    plt.savefig(directory_path + 'weighted_fixation_angle_per_position.png')
    plt.show()


def plot_fixation_distance_per_meta_data(df, subfolder=''):
    directory_path = './imgs/gaze/fixations/fixations_per_position/' + subfolder

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    sns.catplot(data=df, x='experience', y='mfd', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Experience')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_experience.png')
    plt.show()

    sns.catplot(data=df, x='experience', y='mfa', kind='bar')
    plt.suptitle('Average Weighted Fixation Angle per Experience')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_experience.png')
    plt.show()

    sns.catplot(data=df, x='score', y='mfd', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_score.png')
    plt.show()

    sns.catplot(data=df, x='score', y='mfa', kind='bar')
    plt.suptitle('Average Weighted Fixation Angle per Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_score.png')
    plt.show()

    sns.catplot(data=df, x='subject_id', y='mfd', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Subject')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_subject.png')
    plt.show()

    sns.catplot(data=df, x='subject_id', y='mfa', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Subject')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_angle_by_subject.png')
    plt.show()

    sns.catplot(data=df, x='level_score', y='mfd', kind='bar')
    plt.suptitle('Average Weighted Fixation Distance per Level Score')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_level_score.png')
    plt.show()


def plot_mfd_per_level_score(subfolder=''):
    directory_path = './imgs/gaze/fixations/fixations_per_position/' + subfolder

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    fix_df = load_fixations()
    level_scores = pd.read_csv('../data/level_scores.csv')

    mean_mfd_df = fix_df.groupby(['subject_id', 'game_difficulty', 'world_number'])['mfd'].mean().reset_index(
        name='mfd')

    mfd_and_score_df = level_scores.merge(mean_mfd_df, on=['subject_id', 'game_difficulty', 'world_number'], how='left').drop_duplicates()
    mfd_and_score_df = mfd_and_score_df[mfd_and_score_df['level_score'] > 250]  # only won games

    sns.scatterplot(mfd_and_score_df, x='level_score', y='mfd', hue='game_difficulty')
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_level_score.png')
    plt.show()

    sns.scatterplot(mfd_and_score_df, x='standardized_level_score', y='mfd', hue='game_difficulty')
    plt.tight_layout()
    plt.savefig('../thesis/3experts_vs_novices/mfd_by_level_score.png')
    plt.savefig(directory_path + 'weighted_fixation_distance_by_level_score_standardized.png')
    plt.show()

    sns.histplot(mfd_and_score_df, x='mfd', hue='game_difficulty', multiple="stack")
    plt.tight_layout()
    plt.savefig(directory_path + 'weighted_fixation_distance_by_game_difficulty.png')
    plt.show()


def get_region_from_field(y_field):
    if 0 == y_field:
        return 'start'
    if 1 <= y_field <= 6:
        return 'street'
    elif 8 <= y_field <= 13:
        return 'river'
    elif 7 == y_field:
        return 'middle'
    else:
        return np.nan


def plot_fixation_distance_hist_per_region(df):
    directory_path = './imgs/gaze/fixations/fixations_per_position/'

    fig, ax = plt.subplots()
    # multiple{“layer”, “dodge”, “stack”, “fill”}
    n_bins = 20
    x = df[df['region'] == 'street']['mfd']
    ax.hist(x, density=True, bins=n_bins, label='street')
    y = df[df['region'] == 'river']['mfd']
    ax.hist(y, density=True, bins=n_bins, alpha=0.5, label='river')
    ax.set_xscale('log')
    ax.legend()

    # df = df.drop(df[df['mfd'] > 0.3].index)
    # binwidth = 0.04
    # sns.histplot(data=df, ax=ax, x="mfd", hue="region", stat='proportion', multiple='fill', binwidth=binwidth, common_norm=False)  # multiple="stack",
    #
    # ax.set_xlabel('Average Fixation Distance [fields/ms]')
    # plt.savefig(directory_path + 'fixation_distance_per_region_hist.png')
    # plt.savefig('../paper/fixation_distance_per_region_hist.png')
    # plt.savefig('../paper/fixation_distance_per_region_hist.svg', format='svg')
    plt.show()


def plot_mfd_per_region():
    df = load_fixations()
    df = df[(df['region'] == 'street') | (df['region'] == 'river')]
    directory_path = './imgs/gaze/fixations/fixations_per_position/'

    edge_colors = [paper_plot_utils.C0, paper_plot_utils.C1]
    box_colors = [paper_plot_utils.C0_soft, paper_plot_utils.C1_soft]

    # create boxplot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.boxplot(data=df, ax=ax, y="mfd", x="region", width=0.2, linewidth=1.5,
                showfliers=False,
                # flierprops=dict(markersize=2),
                showmeans=True, meanline=True,
                boxprops={'zorder': 2})

    # iterate over boxes
    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    if len(box_patches) == 0:
        box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        patch.set_edgecolor('k')  # edge_colors[i])
        patch.set_facecolor(box_colors[i])

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color('k')  # edge_colors[i])
            line.set_mfc('k')  # edge_colors[i])  # facecolor of fliers
            line.set_mec('k')  # edge_colors[i])  # edgecolor of fliers

    sns.violinplot(data=df, ax=ax, y="mfd", x="region", size=0.5, palette=box_colors, inner=None,
                   saturation=0.5)

    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('Fixation distance [fields]')
    plt.savefig(directory_path + 'fixation_distance_per_region_box.png')
    plt.savefig('../thesis/2river_vs_street/fixation_distance_per_region_box.png')
    plt.savefig('../paper/fixation_distance_per_region_box.svg', format="svg")
    plt.show()


def ttest_fixation_distance_street_river():
    df = load_fixations()

    # get weighted fixation distances
    df_river = get_river_data(df)
    df_street = get_street_data(df)
    print(
        'H0: Mean fixations distances are equal | H1: Mean fixations distances on river are greater than mean fixation distances on street')

    fix_distances_river = df_river['mfd']
    fix_distances_street = df_street['mfd']

    # t test manhattan distances:
    ttest_result = scipy.stats.ttest_ind(fix_distances_river, fix_distances_street, alternative='greater')
    print(ttest_result)
    print('dof=', len(fix_distances_river) - 1 + len(fix_distances_street) - 1)


def kstest_fixation_distance_street_river():
    df = load_fixations()

    # get weighted fixation distances
    df_river = get_river_data(df)
    df_street = get_street_data(df)

    print(
        'H0: Distributions for fixation distances are equal | H1: Distributions for fixation distances are different for river and street')

    fix_distances_river = df_river['mfd']
    fix_distances_street = df_street['mfd']

    # ks test manhattan distances:
    kstest_result = scipy.stats.kstest(fix_distances_river, fix_distances_street, alternative='two-sided')
    print(kstest_result)


def plot_mfd_per_score():
    sns.set_style('whitegrid')
    df = load_fixations()

    order_df = df[['subject_id', 'score']].drop_duplicates().sort_values('score')

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.pointplot(data=df, ax=ax, x='subject_id', y='mfd', join=False, order=order_df['subject_id'])

    xlabels = [subject2letter(subj_id.get_text()) for subj_id in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Subject score')
    ax.set_ylabel('MFD')
    plt.tight_layout()

    plt.savefig('./imgs/gaze/fixations/mfd_per_score.png')
    plt.savefig('../thesis/3experts_vs_novices/mfd_per_score.png')
    plt.show()

    # linear regression:

    # df = df[['subject_id', 'score', 'mfd']].groupby(['subject_id', 'score'])[
    #     'mfd'].mean().reset_index()
    df = df[['subject_id', 'score', 'mfd']].groupby(['subject_id', 'score'])[
        'mfd'].agg(['mean', 'sem', 'std'])
    df.columns = ['mfd', 'mfd_sem', 'mfd_std']
    df.reset_index(inplace=True)

    x = df['score']
    y = df['mfd']
    sem = df['mfd_sem']
    std = df['mfd_std']
    res = scipy.stats.linregress(x, y, alternative='greater')

    print('Linear Regression for MFD per Score:')
    print(f'R-squared: {res.rvalue ** 2}')
    print(f'p value: {res.pvalue}')

    # calc 95% CI for slope
    dof = x.shape[0] - 1
    tinv = lambda p, df: abs(scipy.stats.t.ppf(p / 2, dof))
    ts = tinv(0.05, len(x) - 2)
    slope_ci_upper = res.slope + ts * res.stderr
    slope_ci_lower = res.slope - ts * res.stderr
    intercept_ci_upper = res.intercept + ts * res.intercept_stderr
    intercept_ci_lower = res.intercept - ts * res.intercept_stderr

    # calculate
    steps = 1000
    xlim = (x.min() - 1000, x.max() + 1000)
    xx = np.arange(xlim[0], xlim[1], steps)
    possible_bounds = np.zeros((4, xx.shape[0]))
    for i, (slope, intercept) in enumerate(product([slope_ci_lower, slope_ci_upper], [intercept_ci_lower, intercept_ci_upper])):
        possible_bounds[i] = intercept + xx * slope

    bounds_max = np.max(possible_bounds, axis=0)
    bounds_min = np.min(possible_bounds, axis=0)

    print(f"slope (95%): {res.slope:.6f} +/- {ts * res.stderr:.6f}")
    print(f"intercept (95%): {res.intercept:.6f} +/- {ts * res.intercept_stderr:.6f}")

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    # plt.scatter(x, y, label='data')
    plt.errorbar(x, y, yerr=sem, fmt='o', markersize=2, label='Data')
    plt.plot(xx, res.intercept + res.slope * xx, 'r', label='Linear regression')
    # plt.fill_between(xx, bounds_min, bounds_max, color='r', alpha=0.25, label='95% ci interval')
    plt.xlim(xlim)
    plt.xlabel('Subject score')
    plt.ylabel('MFD')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/mfd_per_score_w_lin_reg.png')
    plt.savefig('../thesis/3experts_vs_novices/mfd_per_score_lin_reg.png')
    plt.show()


def plot_fixation_kde(df, axes):
    if df.shape[0] < 5:
        # TODO check for variance or use try except block
        return

    player_x = int(df['player_x_field'].values[0])
    player_y = int(df['player_y_field'].values[0])

    ax = axes[config.N_LANES - 1 - player_y, player_x]
    ax.set_xlim((0, config.DISPLAY_WIDTH_PX))
    ax.set_ylim((0, config.DISPLAY_HEIGHT_PX))

    sns.kdeplot(df, x='fix_x', y='fix_y', weights='fix_duration', ax=ax, fill=True, cmap='mako')
    sns.scatterplot(df, x='fix_x', y='fix_y', size='fix_duration', ax=ax, alpha=0.5, color='white')

    # add player 2position
    rect = patches.Rectangle((player_x * config.FIELD_WIDTH, player_y * config.FIELD_HEIGHT), config.FIELD_WIDTH, config.FIELD_HEIGHT,
                             fill=True, color='r', alpha=0.5)
    ax.add_patch(rect)


def plot_fixations_kde():
    df = load_fixations()

    fig, axs = plt.subplots(config.N_LANES, config.N_FIELDS_PER_LANE, figsize=(60, 40), sharex='all', sharey='all')

    grp_by_player_pos = df.groupby(['player_x_field', 'player_y_field'])
    grp_by_player_pos.apply(lambda x: plot_fixation_kde(x, axs))

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i, j]
            ax.set_facecolor('darkgray')
            ax.set_xticks(range(0, config.DISPLAY_WIDTH_PX + 1, int(config.FIELD_WIDTH)), rotation=90)
            ax.set_yticks(range(0, config.DISPLAY_HEIGHT_PX + 1, int(config.FIELD_HEIGHT)))
            ax.legend().set_visible(False)
            if i == axs.shape[0] - 1:
                ax.set_xlabel(f'Col {j}')
                ax.set_xticklabels(range(0, config.N_FIELDS_PER_LANE + 1), rotation=90)
            if j == 0:
                ax.set_ylabel(f'Row {config.N_LANES - 1 - i}')
                ax.set_yticklabels(range(0, config.N_LANES + 1))

    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/fix_kde_per_position.png')
    plt.show()


def plot_polar_hist_for_fixations_per_position():
    df = load_fixations()
    fig, axs = plt.subplots(config.N_LANES, config.N_FIELDS_PER_LANE, figsize=(60, 40), subplot_kw=dict(projection="polar"))

    grp_by_player_pos = df.groupby(['player_x_field', 'player_y_field'])
    grp_by_player_pos.apply(lambda x: polar_hist_fixations(x, axs))

    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/fix_polar_hist_per_position.png')
    plt.show()


def polar_hist_fixations(df, axs):
    # get relevant values
    fix_angle = df['fix_angle']
    fix_dist = df['fix_distance_manhattan']
    fix_duration = df['fix_duration']

    # player coordinates
    player_x = int(df['player_x_field'].values[0])
    player_y = int(df['player_y_field'].values[0])

    # get ax
    ax = axs[config.N_LANES - 1 - player_y, player_x]

    # define binning
    angle_bins = np.linspace(0, 2 * np.pi, 45)
    dist_bins = np.linspace(0, 8, 5)  # fix_dist.max(), 5)

    # calculate histogram
    hist, _, _ = np.histogram2d(fix_angle, fix_dist, weights=fix_duration, bins=(angle_bins, dist_bins))
    A, R = np.meshgrid(angle_bins, dist_bins)

    pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r")
    ax.set_theta_offset(np.pi / 2)
    return pc


def calc_weighted_y_distance(game_df):
    total_fix_duration = game_df['fix_duration'].sum()
    y_distance = game_df['fix_y_field'] - game_df['player_y_field']
    return y_distance * game_df['fix_duration'] / total_fix_duration


def plot_gaze_y_position_relative_to_player():
    df = load_fixations()

    df['weighted_y_distance'] = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_weighted_y_distance).values
    ax = sns.histplot(data=df, y='weighted_y_distance', hue='region', stat='proportion', multiple='stack')
    ax.set_ylabel('Weighted fixation distance in y relative to player')
    plt.savefig('./imgs/gaze/fixations/weighted_y_gaze_relative_to_player_hist.png')
    plt.tight_layout()
    plt.show()

    ax = sns.violinplot(data=df, y='weighted_y_distance', x='region')
    ax.set_ylabel('Weighted fixation distance in y relative to player')
    plt.savefig('./imgs/gaze/fixations/weighted_y_gaze_relative_to_player_violin.png')
    plt.tight_layout()
    plt.show()

    print('\n ----------------- Y Fix Position Relative to player')
    max_y = df['weighted_y_distance'].max()
    min_y = df['weighted_y_distance'].min()
    mean_y = df['weighted_y_distance'].mean()
    var_y = df['weighted_y_distance'].var()
    median_y = df['weighted_y_distance'].median()
    print('\nGeneral:')
    print(f'Range: [{min_y:.5f}, {max_y:.5f}] \t| mean {mean_y:.5f} with var {var_y:.5f} \t| median {median_y:.5f}')

    df_street = get_street_data(df)
    max_y_street = df_street['weighted_y_distance'].max()
    min_y_street = df_street['weighted_y_distance'].min()
    mean_y_street = df_street['weighted_y_distance'].mean()
    var_y_street = df_street['weighted_y_distance'].var()
    median_y_street = df_street['weighted_y_distance'].median()
    print('\nStreet:')
    print(
        f'Range: [{min_y_street:.5f}, {max_y_street:.5f}] \t| mean {mean_y_street:.5f} with var {var_y_street:.5f} \t| median {median_y_street:.5f}')
    df_river = get_river_data(df)
    max_y_river = df_river['weighted_y_distance'].max()
    min_y_river = df_river['weighted_y_distance'].min()
    mean_y_river = df_river['weighted_y_distance'].mean()
    var_y_river = df_river['weighted_y_distance'].var()
    median_y_river = df_river['weighted_y_distance'].median()
    print('River:')
    print(
        f'Range: [{min_y_river:.5f}, {max_y_river:.5f}] \t| mean {mean_y_river:.5f} with var {var_y_river:.5f} \t| median {median_y_river:.5f}')

    print('\nt-test:')
    print(
        'H0: Relative Fixation in y is equal for street and river section | H1: Relative Fixation in y on street is greater than relative fixation on river')
    ttest_result = scipy.stats.ttest_ind(df_street['weighted_y_distance'], df_river['weighted_y_distance'], alternative='greater')
    print('Test in Weighted Manhattan Distances')
    print(ttest_result)
    print('dof=', df_street['weighted_y_distance'].shape[0] - 1 + df_river['weighted_y_distance'].shape[0] - 1)


def calc_weighted_x_distance(game_df):
    total_fix_duration = game_df['fix_duration'].sum()
    x_distance = game_df['fix_x_field'] - game_df['player_x_field']
    return x_distance * game_df['fix_duration'] / total_fix_duration


def plot_gaze_x_position_relative_to_player():
    # load data
    df = load_fixations()
    df['weighted_x_distance'] = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_weighted_x_distance).values

    print('\n ----------------- X Fix Position Relative to player')
    max_x = df['weighted_x_distance'].max()
    min_x = df['weighted_x_distance'].min()
    mean_x = df['weighted_x_distance'].mean()
    var_x = df['weighted_x_distance'].var()
    median_x = df['weighted_x_distance'].median()
    print('\nGeneral:')
    print(f'Range: [{min_x:.5f}, {max_x:.5f}] \t| mean {mean_x:.5f} with var {var_x:.5f} \t| median {median_x:.5f}')

    df_street = get_street_data(df)
    max_x_street = df_street['weighted_x_distance'].max()
    min_x_street = df_street['weighted_x_distance'].min()
    mean_x_street = df_street['weighted_x_distance'].mean()
    var_x_street = df_street['weighted_x_distance'].var()
    median_x_street = df_street['weighted_x_distance'].median()
    print('\nStreet:')
    print(
        f'Range: [{min_x_street:.5f}, {max_x_street:.5f}] \t| mean {mean_x_street:.5f} with var {var_x_street:.5f} \t| median {median_x_street:.5f}')
    df_river = get_river_data(df)
    max_x_river = df_river['weighted_x_distance'].max()
    min_x_river = df_river['weighted_x_distance'].min()
    mean_x_river = df_river['weighted_x_distance'].mean()
    var_x_river = df_river['weighted_x_distance'].var()
    median_x_river = df_river['weighted_x_distance'].median()
    print('River:')
    print(
        f'Range: [{min_x_river:.5f}, {max_x_river:.5f}] \t| mean {mean_x_river:.5f} with var {var_x_river:.5f} \t| median {median_x_river:.5f}')

    print('\nt-test:')
    print(
        'H0: Relative Fixation in x is equal for street and river section | H1: Relative Fixation in x on river is greater than relative fixation on street')
    ttest_result = scipy.stats.ttest_ind(df_river['weighted_x_distance'], df_street['weighted_x_distance'], alternative='greater')
    print('Test in Weighted Manhattan Distances')
    print(ttest_result)
    print('dof=', df_street['weighted_x_distance'].shape[0] - 1 + df_river['weighted_x_distance'].shape[0] - 1)

    # TODO this also for target 2position

    # boxplot per lane
    sns.boxplot(x='weighted_x_distance', y='player_y_field', data=df)
    plt.savefig('./imgs/gaze/fixations/weighted_x_gaze_relative_to_player_per_row.png')
    plt.tight_layout()
    plt.show()

    # histogram over x for regions
    ax = sns.histplot(data=df, x='weighted_x_distance', hue='region', stat='proportion', multiple='stack')
    ax.set_xlabel('Weighted fixation distance in x relative to player')
    # ax.set_xlim((-20, 20)) TODO
    plt.savefig('./imgs/gaze/fixations/weighted_x_gaze_relative_to_player_hist.png')
    plt.tight_layout()
    plt.show()

    ax = sns.violinplot(data=df, x='weighted_x_distance', y='region')
    # ax.set_xlim((-20, 20)) TODO
    ax.set_xlabel('Weighted fixation distance in x relative to player')
    plt.savefig('./imgs/gaze/fixations/weighted_x_gaze_relative_to_player_violin.png')
    plt.tight_layout()
    plt.show()


def plot_fixation_KDE_relative_to_player():
    # load data
    df = load_fixations()
    df = df[(df['region'] == 'street') | (df['region'] == 'river')]
    df['weighted_x_distance'] = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_weighted_x_distance).values
    df['weighted_y_distance'] = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_weighted_y_distance).values

    print(df['region'].value_counts())

    lim = 0.2
    palette = {'street': paper_plot_utils.red_kde, 'river': paper_plot_utils.blue_kde}

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    sns.kdeplot(data=df, x="weighted_x_distance", y="weighted_y_distance", hue='region', space=0, palette=palette, ax=ax)
    ax.axhline(0, ls='--', color='k', alpha=0.5)
    ax.axvline(0, ls='--', color='k', alpha=0.5)
    plt.xlim((-lim, lim))
    plt.ylim((-lim, lim))
    plt.xlabel('x')
    plt.ylabel('y')

    # g = sns.JointGrid(data=df, x="weighted_x_distance", y="weighted_y_distance", hue='region', space=0, palette=palette)
    # g.refline(x=0, y=0)
    # g.plot_joint(sns.kdeplot)
    # g.plot_marginals(sns.histplot, binwidth=0.005, multiple='stack')
    # g.ax_marg_x.set_xlim(-lim, lim)
    # g.ax_marg_y.set_ylim(-lim, lim)
    # g.set_axis_labels('x', 'y')

    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/weighted_gaze_position_relative_to_player_kde.png')
    plt.savefig('../thesis/2river_vs_street/2kde/weighted_gaze_position_relative_to_player_kde.png')
    plt.show()
    #
    # # do it for every target position
    # g = sns.displot(data=df, x='weighted_x_distance', y='weighted_y_distance', hue='region', kind='kde', col='target_position')
    # plt.tight_layout()
    #
    # plt.savefig('./imgs/gaze/fixations/weighted_gaze_position_relative_to_player_kde_per_target_position.png')
    # plt.show()


def plot_weighted_fixations_relative_to_player_per_fixated_object():
    # load data
    all_df = load_fixations()
    all_df['weighted_x_distance'] = all_df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_weighted_x_distance).values
    all_df['weighted_y_distance'] = all_df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_weighted_y_distance).values
    df = all_df[(all_df['region'] == 'street') | (all_df['region'] == 'river')]
    df = df[df['fixation_on'] == 'world'].copy()

    lim = 0.35
    g = sns.JointGrid(data=df, x="weighted_x_distance", y="weighted_y_distance", hue='region', space=0)
    g.refline(x=0, y=0)
    g.plot_joint(sns.kdeplot, levels=15)
    g.plot_marginals(sns.histplot, binwidth=0.005, multiple='stack')
    g.ax_marg_x.set_xlim(-lim, lim)
    g.ax_marg_y.set_ylim(-lim, lim)
    g.set_axis_labels('x', 'y')

    plt.suptitle('fix on world only')
    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/weighted_gaze_position_relative_to_player_kde_world_only.png')
    plt.show()

    df = all_df[(all_df['fixation_on'] == 'near_target') | (all_df['fixation_on'] == 'target')].copy()
    lim = 0.5
    g = sns.JointGrid(data=df, x="weighted_x_distance", y="weighted_y_distance", hue='region', space=0)
    g.refline(x=0, y=0)
    g.plot_joint(sns.kdeplot, levels=15)
    g.plot_marginals(sns.histplot, binwidth=0.005, multiple='stack')
    g.ax_marg_x.set_xlim(-lim, lim)
    g.ax_marg_y.set_ylim(-lim, lim)
    g.set_axis_labels('x', 'y')

    plt.suptitle('fix on target only')
    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/weighted_gaze_position_relative_to_player_kde_target_only.png')
    plt.show()

    df = all_df[(all_df['fixation_on'] == 'near_player') | (all_df['fixation_on'] == 'player')].copy()
    df = df[(df['region'] == 'street') | (df['region'] == 'river')]

    lim = 0.15
    g = sns.JointGrid(data=df, x="weighted_x_distance", y="weighted_y_distance", hue='region', space=0)
    g.refline(x=0, y=0)
    g.plot_joint(sns.kdeplot, levels=15)
    g.plot_marginals(sns.histplot, binwidth=0.005, multiple='stack')
    g.ax_marg_x.set_xlim(-lim, lim)
    g.ax_marg_y.set_ylim(-lim, lim)
    g.set_axis_labels('x', 'y')

    plt.suptitle('fix on player only')
    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/weighted_gaze_position_relative_to_player_kde_player_only.png')
    plt.show()

    g = sns.displot(data=all_df, x="weighted_x_distance", y="weighted_y_distance", col='region', kind="kde", col_wrap=2)
    g.refline(x=0, y=0)
    # TODO format plot (lims for each ax)
    plt.xlim((-0.25, 0.25))

    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/weighted_gaze_position_relative_to_player_kde_per_region.png')
    plt.show()


def print_fixations_on_target_for_region():
    df = load_fixations()
    street_df = df[df['region'] == 'street']
    river_df = df[df['region'] == 'river']

    print('\nNumber | Ratio of fixations on target in street section:')
    target_street_df = street_df[(street_df['fixation_on'] == 'near_target') | (street_df['fixation_on'] == 'target')]
    n_all_street = street_df.shape[0]
    n_target_street = target_street_df.shape[0]
    print(f'{n_target_street}/{n_all_street} | {n_target_street / n_all_street * 100} %')

    print('Number | Ratio of fixations on target in river section:')
    target_river_df = river_df[(river_df['fixation_on'] == 'near_target') | (river_df['fixation_on'] == 'target')]
    n_all_river = river_df.shape[0]
    n_target_river = target_river_df.shape[0]
    print(f'{n_target_river}/{n_all_river} | {n_target_river / n_all_river * 100} %')

    print('\nAverage MFD when fixating on target in river section:')
    river_target_mfd = target_river_df["mfd"]
    print(f'{river_target_mfd.mean()} with var {river_target_mfd.var()} | median={river_target_mfd.median()}')
    print(f'Range: {river_target_mfd.min()} to {river_target_mfd.max()}')

    print(f'\nTarget Fixations per Lane:')
    print(df[(df['fixation_on'] == 'near_target') | (df['fixation_on'] == 'target')]['player_y_field'].value_counts().sort_index(
        ascending=False))

    print('\nFixations when in middle lane:')
    print(df[df['region'] == 'middle']['fixation_on'].value_counts())

    print('\nFixations when in start lane:')
    print(df[df['region'] == 'start']['fixation_on'].value_counts())


def plot_fixations_on_target_per_lane():
    df = load_fixations()
    target_fixations = df[(df['fixation_on'] == 'near_target') | (df['fixation_on'] == 'target')]

    palette = {'near_target': paper_plot_utils.blue, 'target': paper_plot_utils.red}
    ax = sns.histplot(target_fixations, y='player_y_field', hue='fixation_on', discrete=True, multiple='stack', stat='proportion',
                      palette=palette)

    y_lim_lower_offset = 0 - ax.get_ylim()[0]
    ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1] + y_lim_lower_offset))

    plt.ylabel('y')
    plt.xlabel('Proportion of fixations on target')
    plt.legend(title='Fixation', loc='lower right', labels=['On target', 'Near target'], framealpha=1.0)

    # add text
    x_lim = ax.get_xlim()[1]
    plt.axhline(y=7, color='black', linestyle="dashed", alpha=0.5)
    plt.text(y=7, x=x_lim / 2, s='middle lane', ha='center', va='center', backgroundcolor='white', alpha=0.5,
             bbox=dict(pad=1.5, facecolor='white', edgecolor='white'))

    plt.axhline(y=0, color='black', linestyle="dashed", alpha=0.5)
    plt.text(y=0, x=x_lim / 2, s='start lane', ha='center', va='center', backgroundcolor='white', alpha=0.5,
             bbox=dict(pad=1.5, facecolor='white', edgecolor='white'))

    plt.axhline(y=14, color='black', linestyle="dashed", alpha=0.5)
    plt.text(y=14, x=x_lim / 2, s='finish lane', ha='center', va='center', backgroundcolor='white', alpha=0.5,
             bbox=dict(pad=1.5, facecolor='white', edgecolor='white'))

    plt.savefig('../thesis/1descriptive/3gaze/fixations_on_target.png')
    plt.show()


def plot_MFD_diff_river_street_over_score():
    df = load_fixations()

    mfd_river = df[df['region'] == 'river'].groupby('subject_id')['mfd'].mean()
    mfd_street = df[df['region'] == 'street'].groupby('subject_id')['mfd'].mean()
    mfd_diff = (mfd_river - mfd_street).reset_index().rename(columns={'mfd': 'mfd_diff'})
    mfd_diff = mfd_diff.merge(df[['subject_id', 'score']].drop_duplicates(), on='subject_id', how='left')

    order_df = df[['subject_id', 'score']].drop_duplicates().sort_values('score')

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.pointplot(data=mfd_diff, ax=ax, x='subject_id', y='mfd_diff', join=False, order=order_df['subject_id'])

    xlabels = [subject2letter(subj_id.get_text()) for subj_id in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    plt.tight_layout()

    plt.savefig('./imgs/gaze/fixations/mfd_diff_per_score.png')
    plt.savefig('../thesis/3experts_vs_novices/mfd_diff_per_score.png')
    plt.show()

    # linear regression:

    mfd_diff = mfd_diff[['subject_id', 'score', 'mfd_diff']].groupby(['subject_id', 'score'])[
        'mfd_diff'].agg(['mean', 'sem', 'std'])
    mfd_diff.columns = ['mfd_diff', 'mfd_diff_sem', 'mfd_diff_std']
    mfd_diff.reset_index(inplace=True)

    x = mfd_diff['score']
    y = mfd_diff['mfd_diff']
    sem = mfd_diff['mfd_diff_sem']
    std = mfd_diff['mfd_diff_std']
    res = scipy.stats.linregress(x, y, alternative='greater')

    print('Linear Regression for MFD diff per score:')
    print(f'R-squared: {res.rvalue ** 2}')
    print(f'p value: {res.pvalue}')

    # calc 95% CI for slope
    dof = x.shape[0] - 1
    tinv = lambda p, df: abs(scipy.stats.t.ppf(p / 2, dof))
    ts = tinv(0.05, len(x) - 2)
    slope_ci_upper = res.slope + ts * res.stderr
    slope_ci_lower = res.slope - ts * res.stderr
    intercept_ci_upper = res.intercept + ts * res.intercept_stderr
    intercept_ci_lower = res.intercept - ts * res.intercept_stderr

    # calculate
    steps = 1000
    xlim = (x.min() - 1000, x.max() + 1000)
    xx = np.arange(xlim[0], xlim[1], steps)
    possible_bounds = np.zeros((4, xx.shape[0]))
    for i, (slope, intercept) in enumerate(product([slope_ci_lower, slope_ci_upper], [intercept_ci_lower, intercept_ci_upper])):
        possible_bounds[i] = intercept + xx * slope

    bounds_max = np.max(possible_bounds, axis=0)
    bounds_min = np.min(possible_bounds, axis=0)

    print(f"slope (95%): {res.slope:.6f} +/- {ts * res.stderr:.6f}")
    print(f"intercept (95%): {res.intercept:.6f} +/- {ts * res.intercept_stderr:.6f}")

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    # plt.scatter(x, y, label='data')
    plt.errorbar(x, y, yerr=sem, fmt='o', markersize=2, label='Data')
    plt.plot(xx, res.intercept + res.slope * xx, 'r', label='Linear regression')
    # plt.fill_between(xx, bounds_min, bounds_max, color='r', alpha=0.25, label='95% ci interval')
    plt.xlim(xlim)
    plt.xlabel('Subject score')
    plt.ylabel('MFD delta river to street')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./imgs/gaze/fixations/mfd_diff_per_score_w_lin_reg.png')
    plt.savefig('../thesis/3experts_vs_novices/mfd_diff_per_score_lin_reg.png')
    plt.show()


def print_n_fixations_per_region():
    df = load_fixations()
    print(df['region'].value_counts())
    print('In total:', df.shape[0])


if __name__ == '__main__':
    plot_mfd_per_score()
    plot_mfd_heatmap()

    plot_fixation_KDE_relative_to_player()
    print_n_fixations_per_region()
    plot_fixation_heatmap()
    plot_fixations_on_target_per_lane()
    # plot_avg_fixation_distance_per_subject()
    #
    # plot_fixations_kde()

    # plot_polar_hist_for_fixations_per_position()

    # plot_mfd_per_level_score()

    # df = load_fixations()
    # plot_fixation_angle_per_position(df)
    # plot_fixation_distance_per_position(df)
    # plot_fixation_distance_box_per_region(df)

    plot_MFD_diff_river_street_over_score()

    # plot_gaze_y_position_relative_to_player()
    print_fixations_on_target_for_region()
    plot_gaze_x_position_relative_to_player()

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
