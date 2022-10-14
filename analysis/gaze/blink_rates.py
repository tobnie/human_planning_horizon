from itertools import product

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns

from analysis import paper_plot_utils
from analysis.data_utils import get_river_data, get_street_data, read_data


def calc_blink_rate_per_game(game_df):
    """ Calculate blink rates per game and for street and river section. If a section was not entered, the blink rate is assigned np.nan"""
    game_df['time_delta'] = game_df['time'].diff()

    street_df = get_street_data(game_df)
    river_df = get_river_data(game_df)

    # game time and times on areas
    # remember that game_time >= time_on_street + time_on_river since there are also start and middle lane
    # also convert from msec to sec
    game_time = game_df['time'].max() * 1e-3
    time_on_street = street_df['time_delta'].sum() * 1e-3
    time_on_river = river_df['time_delta'].sum() * 1e-3

    # number of blinks
    n_blinks_street = len(street_df[street_df['blink_start'].notna()])
    n_blinks_river = len(river_df[river_df['blink_start'].notna()])
    n_blinks_total = len(game_df[game_df['blink_start'].notna()])

    # calculate blink rates
    blink_rate_street = n_blinks_street / time_on_street if time_on_street > 0 else np.nan
    blink_rate_river = n_blinks_river / time_on_river if time_on_river > 0 else np.nan
    blink_rate_total = n_blinks_total / game_time if n_blinks_total > 0 else np.nan

    return blink_rate_total, blink_rate_street, blink_rate_river


def save_blink_rates():
    blinks = pd.read_csv('../data/blinks.csv')
    df = read_data()

    joined_df = df.merge(blinks, left_on=['subject_id', 'game_difficulty', 'world_number', 'time'],
                         right_on=['subject_id', 'game_difficulty', 'world_number', 'blink_start'], how='left')
    blink_rate_info = joined_df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_blink_rate_per_game).reset_index(
        name='blink_info')
    blink_info = pd.DataFrame(blink_rate_info['blink_info'].to_list(), columns=['blink_rate', 'blink_rate_street', 'blink_rate_river'])
    blink_rates = pd.concat([blink_rate_info[['subject_id', 'game_difficulty', 'world_number']], blink_info], axis=1)
    blink_rates.to_csv('../data/blink_rates.csv', index=False)
    return blink_rates


def plot_blink_rates():
    blink_rates = pd.read_csv('../data/blink_rates.csv')

    blink_rates_total = blink_rates['blink_rate'].dropna()
    blink_rates_street = blink_rates['blink_rate_street'].dropna()
    blink_rates_river = blink_rates['blink_rate_river'].dropna()

    # blink rates everywhere
    print('--- Blink Rate Total ---')
    print('Mean: ', blink_rates_total.mean())
    print('Median: ', blink_rates_total.median())
    print('Variance: ', blink_rates_total.var())

    # blink rates on street
    print('--- Street ---')
    print('Mean: ', blink_rates_street.mean())
    print('Median: ', blink_rates_street.median())
    print('Variance: ', blink_rates_street.var())

    # blink rates on river
    print('--- River ---')
    print('Mean: ', blink_rates_river.mean())
    print('Median: ', blink_rates_river.median())
    print('Variance: ', blink_rates_river.var())

    plt.boxplot([blink_rates_total.values, blink_rates_street.values, blink_rates_river.values], labels=['total', 'street', 'river'])
    plt.ylabel('blink rate [blinks/s]')
    plt.savefig('./imgs/blinks/blink_rates_box.png')
    plt.savefig('../thesis/2river_vs_street/3blink_rate/blink_rates_box.png')
    plt.show()

    # fancy blink rate plot (box plot and violin plot)
    # TODO
    df = df[(df['region'] == 'street') | (df['region'] == 'river')]

    edge_colors = [paper_plot_utils.C0, paper_plot_utils.C1]
    box_colors = [paper_plot_utils.C0_soft, paper_plot_utils.C1_soft]

    # create boxplot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.boxplot(data=df, ax=ax, y="pupil_size_z", x="region", width=0.2, linewidth=1.5,
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

    sns.violinplot(data=df, ax=ax, y="pupil_size_z", x="region", size=0.5, scale='width', palette=box_colors, inner=None,
                   saturation=0.5)

    ax.set_xlabel('')
    ax.set_ylabel('Pupil size z-score')
    plt.savefig('./imgs/gaze/fixations/fixations_per_position/fixation_distance_per_region_box.png')
    plt.savefig('../thesis/2river_vs_street/pupil_size_per_region_box.png')
    plt.show()


def ttest_blink_rate_street_river():
    blink_rates = pd.read_csv('../data/blink_rates.csv')
    blink_rates_street = blink_rates['blink_rate_street'].dropna().values
    blink_rates_river = blink_rates['blink_rate_river'].dropna().values

    print('H0: Blink rates are equal | H1: Blink rates on river are greater than blink rates on street')

    # perform (Welch's) t-test
    # t test euclidean distances:
    ttest_result = scipy.stats.ttest_ind(blink_rates_river, blink_rates_street,  alternative='greater')
    print(ttest_result)
    print('dof=', len(blink_rates_street) - 1 + len(blink_rates_river) - 1)


def kstest_blink_rate_distance_street_river():
    blink_rates = pd.read_csv('../data/blink_rates.csv')
    blink_rates_street = blink_rates['blink_rate_street'].dropna().values
    blink_rates_river = blink_rates['blink_rate_river'].dropna().values

    print('H0: Distributions for blink rates are equal | H1: Distributions for blink rates are different for street and river')

    # perform (Welch's) t-test
    # t test euclidean distances:
    kstest_result = scipy.stats.kstest(blink_rates_river, blink_rates_street, alternative='two-sided')
    print(kstest_result)


def plot_blink_rate_over_score():
    blink_rates = pd.read_csv('../data/blink_rates.csv')
    scores = pd.read_csv('../data/level_scores.csv')
    df = blink_rates.merge(scores, on=['subject_id', 'game_difficulty', 'world_number'], how='left')

    # linear regression:
    df = df[['subject_id', 'score', 'blink_rate']].groupby(['subject_id', 'score'])[
        'blink_rate'].agg(['mean', 'sem', 'std'])
    df.columns = ['blink_rate', 'br_sem', 'br_std']
    df.reset_index(inplace=True)

    x = df['score']
    y = df['blink_rate']
    sem = df['br_sem']
    std = df['br_std']
    res = scipy.stats.linregress(x, y, alternative='less')

    print('Linear Regression for avg blinking rate per score:')
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
    plt.errorbar(x, y, yerr=sem, fmt='o', markersize=2, label='data')
    plt.plot(xx, res.intercept + res.slope * xx, 'r', label='linReg')
    # plt.fill_between(xx, bounds_min, bounds_max, color='r', alpha=0.25, label='95% ci interval')
    plt.xlim(xlim)
    plt.xlabel('score')
    plt.ylabel('blinking rate [$s^{-1}$]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./imgs/blinks/blink_rate_per_score_w_lin_reg.png')
    plt.savefig('../thesis/3experts_vs_novices/blink_rate_per_score_w_lin_reg.png')
    plt.show()


def plot_blink_rate_over_level_score():
    blink_rates = pd.read_csv('../data/blink_rates.csv')
    scores = pd.read_csv('../data/level_scores.csv')
    df = blink_rates.merge(scores, on=['subject_id', 'game_difficulty', 'world_number'], how='left')

    # only won games
    df = df[df['standardized_level_score'] > 200]

    # remove nan blink rates
    df = df.dropna(subset=['blink_rate'])

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    # plt.scatter(x, y, label='data')
    sns.scatterplot(df, x='standardized_level_score', y='blink_rate', hue='subject_id', ax=ax)   # TODO investigate exponential relation? --> single scatter plots for each subject?
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig('./imgs/blinks/blink_rate_per_level_score_w_lin_reg_per_subject.png')
    plt.show()

    # linear regression:
    x = df['standardized_level_score']
    y = df['blink_rate']
    res = scipy.stats.linregress(x, y, alternative='greater')

    print('\n\nLinear Regression for avg blinking rate per standardized level score:')
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
    xlim = (x.min() - 10, x.max() + 10)
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
    sns.scatterplot(df, x='standardized_level_score', y='blink_rate', ax=ax)
    ax.plot(xx, res.intercept + res.slope * xx, 'r', label='linear regression')
    # ax.fill_between(xx, bounds_min, bounds_max, color='r', alpha=0.25, label='95% ci interval')
    ax.set_xlabel('level score without difficulty multipliers')
    ax.set_ylabel('blink rate')
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig('./imgs/blinks/blink_rate_per_level_score_w_lin_reg.png')
    plt.savefig('../thesis/3experts_vs_novices/blink_rate_per_level_score_w_lin_reg.png')
    plt.show()


if __name__ == '__main__':
    # save_blink_rates()
    plot_blink_rate_over_score()
    plot_blink_rate_over_level_score()
    plot_blink_rates()
    ttest_blink_rate_street_river()
    kstest_blink_rate_distance_street_river()
