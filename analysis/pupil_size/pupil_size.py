from itertools import product

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

from analysis import paper_plot_utils
from analysis.data_utils import read_data, subject2letter


def get_nan_idx(df, expand=2):
    pupil_sizes = df['pupil_size']
    idx = list(df[pupil_sizes == 0].index)

    # return no indices if there are
    if len(idx) == 0:
        return []

    # expand indices by 2 in each direction to also drop samples near a blink
    prev_idx = idx[0]
    expanded_idx = [prev_idx]

    # add extra samples before first index
    for i in range(expand):
        shift = i + 1
        if prev_idx - shift >= pupil_sizes.index.min():
            expanded_idx.append(prev_idx - shift)

    for i in idx[1:]:
        expanded_idx.append(i)

        if i - prev_idx > 1:
            expanded_idx.append(prev_idx + 1)
            expanded_idx.append(prev_idx + 2)
            expanded_idx.append(i - 2)
            expanded_idx.append(i - 1)

        # iterate
        prev_idx = i

    # add extra samples after last index
    last_index = idx[-1]
    for i in range(expand):
        shift = i + 1
        if last_index + shift < pupil_sizes.index.max():
            expanded_idx.append(last_index + shift)

    return expanded_idx


def set_gaze_information_nan(game_df):
    expand_around_na_samples = 2
    idx = get_nan_idx(game_df, expand_around_na_samples)
    if len(idx) > 0:
        game_df['pupil_size'][idx] = np.nan
    return game_df


def add_pupil_size_z_score(subject_df):
    # drop zero pupil size and samples around them
    subject_df = subject_df.groupby(['subject_id', 'game_difficulty', 'world_number'], group_keys=False).apply(
        set_gaze_information_nan).reset_index()

    # standardize for subject
    pupil_size = subject_df['pupil_size']
    subject_df['pupil_size_z'] = (pupil_size - pupil_size.mean()) / pupil_size.std()
    return subject_df


def plot_pupil_size_over_time_per_game(game_df):
    game_df.dropna(subset='pupil_size_z', inplace=True)
    subject_id = game_df['subject_id'].values[0]
    game_difficulty = game_df['game_difficulty'].values[0]
    world_number = game_df['world_number'].values[0]
    plt.plot(game_df['time'], game_df['pupil_size_z'])
    file_name = 'pupil_size_{}_{}_{}.png'.format(subject_id, game_difficulty, world_number)
    plt.title(file_name)
    plt.xlabel('time [ms]')
    plt.ylabel('pupil size [mm^2], z-standardized')
    plt.savefig('./imgs/pupil_size/per_subject_over_time/' + file_name)
    plt.show()


def plot_pupil_size_for_each_game():
    df = read_data()
    df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(plot_pupil_size_over_time_per_game)


def plot_pupil_size():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')
    df = df[df['pupil_size_z'] < 15]

    # pupil size everywhere
    pupil_size_total = df['pupil_size_z']
    print('--- Pupil Size Total ---')
    print('Mean: ', pupil_size_total.mean())
    print('Median: ', pupil_size_total.median())
    print('Variance: ', pupil_size_total.var())

    # pupil size on street
    street_mask = df['region'] == 'street'
    pupil_size_street = df[street_mask]['pupil_size_z']

    print('--- Street ---')
    print('Mean: ', pupil_size_street.mean())
    print('Median: ', pupil_size_street.median())
    print('Variance: ', pupil_size_street.var())

    # pupil size on river
    river_mask = df['region'] == 'river'
    pupil_size_river = df[river_mask]['pupil_size_z']
    print('--- River ---')
    print('Mean: ', pupil_size_river.mean())
    print('Median: ', pupil_size_river.median())
    print('Variance: ', pupil_size_river.var())

    plt.boxplot([pupil_size_total, pupil_size_street, pupil_size_river], labels=['total', 'street', 'river'])
    plt.ylabel('pupil size z-score')
    plt.savefig('./imgs/pupil_size/pupil_sizes_box.png')
    plt.savefig('../thesis/2river_vs_street/4pupil_size/pupil_sizes_box.png')
    plt.show()

    plt.violinplot(dataset=[pupil_size_total, pupil_size_street, pupil_size_river],
                   showextrema=False)  # , positions=['total', 'street', 'river'])
    plt.ylabel('pupil size z-score')
    plt.ylim((-3, 5))
    plt.savefig('./imgs/pupil_size/pupil_sizes_violin.png')
    plt.savefig('../thesis/2river_vs_street/4pupil_size/pupil_sizes_violin.png')
    plt.show()


def plot_pupil_size_per_region():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')
    df = df[df['pupil_size_z'] < 15]

    # fancy pupil size plot (box plot and violin plot)
    df = df[(df['region'] == 'street') | (df['region'] == 'river')]
    print(df['region'].value_counts())

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

    sns.violinplot(data=df, ax=ax, y="pupil_size_z", x="region", size=0.5, palette=box_colors, inner=None,
                   saturation=0.5)

    ax.set_xlabel('')
    ax.set_ylabel('Pupil size z-score')
    plt.savefig('./imgs/gaze/fixations/fixations_per_position/fixation_distance_per_region_box.png')
    plt.savefig('../thesis/2river_vs_street/4pupil_size/pupil_size_per_region_box.png')
    plt.show()


def ttest_pupil_size_street_river():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')
    df = df[df['pupil_size_z'] < 15]

    # pupil size on street
    street_mask = df['region'] == 'street'
    pupil_size_street = df[street_mask]['pupil_size_z']

    # pupil size on river
    river_mask = df['region'] == 'river'
    pupil_size_river = df[river_mask]['pupil_size_z']

    # t test var 1
    print('H0: Pupil sizes are equal | H1: Pupil sizes on street are greater than pupil sizes on river')
    ttest_result = scipy.stats.ttest_ind(pupil_size_street, pupil_size_river, alternative='greater')
    print(ttest_result)
    print('dof=', len(pupil_size_street) - 1 + len(pupil_size_river) - 1)


def kstest_pupil_size_street_river():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')
    df = df[df['pupil_size_z'] < 15]

    # pupil size on street
    street_mask = df['region'] == 'street'
    pupil_size_street = df[street_mask]['pupil_size_z']

    # pupil size on river
    river_mask = df['region'] == 'river'
    pupil_size_river = df[river_mask]['pupil_size_z']
    print('H0: Distributions for pupil sizes are equal | H1: Distributions for pupil sizes are different for street and river')

    # kstest
    kstest_result = scipy.stats.kstest(pupil_size_river, pupil_size_street, alternative='two-sided')
    print(kstest_result)


def normalize_pupil_size_subject(subject_df):
    return (subject_df - subject_df.min()) / (subject_df.max() - subject_df.min())


def plot_pupil_size_over_score():
    df = read_data()
    df = df[['subject_id', 'score', 'pupil_size', 'pupil_size_z']].dropna(subset=['pupil_size', 'pupil_size_z'])
    df = df[df['pupil_size_z'] < 15]

    # normalize pupil size between 0 and 1
    df['pupil_size_normalized'] = df.groupby(['subject_id', 'score'])['pupil_size'].apply(normalize_pupil_size_subject)

    order_df = df[['subject_id', 'score']].drop_duplicates().sort_values('score')

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.pointplot(data=df, ax=ax, x='subject_id', y='pupil_size_normalized', join=False, order=order_df['subject_id'])

    xlabels = [subject2letter(subj_id.get_text()) for subj_id in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Subject score')
    ax.set_ylabel('Normalized pupil size ')
    plt.tight_layout()
    plt.savefig('./imgs/pupil_size/pupil_size_over_score.png')
    plt.savefig('../thesis/3experts_vs_novices/pupil_size_over_score.png')
    plt.show()

    # linear regression:
    df = df[['subject_id', 'score', 'pupil_size_normalized']].groupby(['subject_id', 'score'])[
        'pupil_size_normalized'].agg(['mean', 'sem', 'std'])
    df.columns = ['pupil_size', 'pupil_size_sem', 'pupil_size_std']
    df.reset_index(inplace=True)

    x = df['score']
    y = df['pupil_size']
    sem = df['pupil_size_sem']
    std = df['pupil_size_std']
    res = scipy.stats.linregress(x, y, alternative='two-sided')

    print('Linear Regression for pupil size per score:')
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

    plt.errorbar(x, y, yerr=sem, fmt='o', markersize=2, label='Data')
    plt.plot(xx, res.intercept + res.slope * xx, 'r', label='Linear regression')
    # plt.fill_between(xx, bounds_min, bounds_max, color='r', alpha=0.25, label='95% ci interval')
    plt.xlim(xlim)
    plt.xlabel('Subject score')
    plt.ylabel('Normalized pupil size')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./imgs/pupil_size/pupil_size_over_score_w_lin_reg.png')
    plt.savefig('../thesis/3experts_vs_novices/pupil_size_over_score_w_lin_reg.png')
    plt.show()


if __name__ == '__main__':
    plot_pupil_size_over_score()
    ttest_pupil_size_street_river()
    kstest_pupil_size_street_river()
    plot_pupil_size()
    plot_pupil_size_per_region()
    plot_pupil_size_for_each_game()
