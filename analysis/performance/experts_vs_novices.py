import pandas as pd
import seaborn as sns
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl

from analysis import paper_plot_utils
from analysis.data_utils import get_all_subjects


def get_fixations_with_scores():
    return pd.read_csv('../data/fixations.csv')


def add_scoring_group_information_to_df(df, percent_split=0.25):
    performance_stats = pd.read_csv('../data/performance_stats.csv', index_col=0)

    scores_only = performance_stats[['subject_id', 'score']].drop_duplicates()

    q025 = scores_only['score'].quantile(percent_split)

    print(f'Scoring groups split at {q025} which is {100 * percent_split}% percentile')
    # classify based on score
    scores_only['scoring_group'] = scores_only['score'].apply(lambda x: 'low' if x < q025 else 'high')
    print('Split Ratio {}/{}'.format(scores_only[scores_only['scoring_group'] == 'low'].shape[0],
                                     scores_only[scores_only['scoring_group'] == 'high'].shape[0]))

    df['scoring_group'] = df['score'].apply(lambda x: 'low' if x < q025 else 'high')

    return df


def print_fixation_distances_per_group():
    df = get_fixations_with_scores()
    df = add_scoring_group_information_to_df(df)

    print('Manhattan Distance')
    print('-----High Scorers-----')
    df_high_scorers = df[df['scoring_group'] == 'high']
    print('n =', len(df_high_scorers['subject_id'].unique()))
    print('Mean:', df_high_scorers['weighted_fix_distance_manhattan'].mean())
    print('Variance:', df_high_scorers['weighted_fix_distance_manhattan'].var())
    print('Median:', df_high_scorers['weighted_fix_distance_manhattan'].median())

    print('-----Low Scorers-----')
    df_low_scorers = df[df['scoring_group'] == 'low']
    print('n =', len(df_low_scorers['subject_id'].unique()))
    print('Mean:', df_low_scorers['weighted_fix_distance_manhattan'].mean())
    print('Variance:', df_low_scorers['weighted_fix_distance_manhattan'].var())
    print('Median:', df_low_scorers['weighted_fix_distance_manhattan'].median())

    print('\n\nEuclidean Distance')
    print('-----High Scorers-----')
    df_high_scorers = df[df['scoring_group'] == 'high']
    print('n =', len(df_high_scorers['subject_id'].unique()))
    print('Mean:', df_high_scorers['weighted_fix_distance_euclidean'].mean())
    print('Variance:', df_high_scorers['weighted_fix_distance_euclidean'].var())
    print('Median:', df_high_scorers['weighted_fix_distance_euclidean'].median())

    print('-----Low Scorers-----')
    df_low_scorers = df[df['scoring_group'] == 'low']
    print('n =', len(df_low_scorers['subject_id'].unique()))
    print('Mean:', df_low_scorers['weighted_fix_distance_euclidean'].mean())
    print('Variance:', df_low_scorers['weighted_fix_distance_euclidean'].var())
    print('Median:', df_low_scorers['weighted_fix_distance_euclidean'].median())


def plot_fixation_distance_box_scoring_groups():
    # get data
    df = get_fixations_with_scores()
    df = add_scoring_group_information_to_df(df)

    edge_colors = [paper_plot_utils.C0, paper_plot_utils.C1]
    box_colors = [paper_plot_utils.C0_soft, paper_plot_utils.C1_soft]

    # create boxplot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    sns.boxplot(data=df, ax=ax, x='scoring_group', y='weighted_fix_distance_manhattan', width=0.2, linewidth=1.5,
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
    ax.set_xlabel('Scoring group')
    ax.set_ylabel('Fixation distance [fields/ms]')
    # ax.set_ylim((1e-4, 1))

    plt.savefig('../paper/fixation_distance_scoring_group.svg', format='svg')
    plt.show()


def plot_fixation_distance_scoring_groups_for_different_splits():
    # get data
    df = get_fixations_with_scores()
    n = len(get_all_subjects())
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))

    for i in range(n - 1):
        ax = axs[int(i // 4), i % 4]
        percent_split = (i + 1) / n
        df_ax = add_scoring_group_information_to_df(df, percent_split)

        edge_colors = [paper_plot_utils.C0, paper_plot_utils.C1]
        box_colors = [paper_plot_utils.C0_soft, paper_plot_utils.C1_soft]

        # create boxplot
        sns.set_style("whitegrid")

        sns.boxplot(data=df_ax, ax=ax, x='scoring_group', y='weighted_fix_distance_manhattan', width=0.2, linewidth=1.5,
                    flierprops=dict(markersize=2),
                    showmeans=True, meanline=True)

        # iterate over boxes
        box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
        if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
            box_patches = ax.artists
        num_patches = len(box_patches)
        lines_per_boxplot = len(ax.lines) // num_patches
        for j, patch in enumerate(box_patches):
            # Set the linecolor on the patch to the facecolor, and set the facecolor to None
            patch.set_edgecolor(edge_colors[j])
            patch.set_facecolor(box_colors[j])

            # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same color as above
            for line in ax.lines[j * lines_per_boxplot: (j + 1) * lines_per_boxplot]:
                line.set_color(edge_colors[j])
                line.set_mfc(edge_colors[j])  # facecolor of fliers
                line.set_mec(edge_colors[j])  # edgecolor of fliers

        ax.set_title('{}/{} split'.format(i + 1, n - (i + 1)))
        ax.set_yscale('log')
        ax.set_xlabel('Scoring group')
        ax.set_ylabel('Fixation distance [fields/ms]')
        # ax.set_ylim((1e-4, 1))

    # plt.savefig('../paper/fixation_distance_scoring_group.svg', format='svg')
    plt.tight_layout()
    plt.show()


def ttest_fixation_distance_scoring_groups():
    # get data
    df = get_fixations_with_scores()
    df = add_scoring_group_information_to_df(df)

    # get weighted fixation distances
    df_high_scorers = df[df['scoring_group'] == 'high']
    df_low_scorers = df[df['scoring_group'] == 'low']
    fix_distances_high_scorers = df_high_scorers['weighted_fix_distance_euclidean']
    fix_distances_low_scorers = df_low_scorers['weighted_fix_distance_euclidean']

    print('H0: Same Means | H1: Mean Weighted Distance for Low Scorers less than for High Scorers')

    # perform (Welch's) t-test
    # t test euclidean distances:
    ttest_result = scipy.stats.ttest_ind(fix_distances_low_scorers, fix_distances_high_scorers,
                                         alternative='less')  # use equal_var=False bc of different sample sizes
    print('Test in Weighted Euclidean Distances')
    print(ttest_result)
    print('dof=', len(fix_distances_low_scorers) - 1 + len(fix_distances_high_scorers) - 1)

    fix_distances_high_scorers = df_high_scorers['weighted_fix_distance_manhattan']
    fix_distances_low_scorers = df_low_scorers['weighted_fix_distance_manhattan']

    # perform (Welch's) t-test
    # t test manhattan distances:
    ttest_result = scipy.stats.ttest_ind(fix_distances_low_scorers, fix_distances_high_scorers,
                                         alternative='less')  # use equal_var=False bc of different sample sizes
    print('Test in Weighted Manhattan Distances')
    print(ttest_result)
    print('dof=', len(fix_distances_low_scorers) - 1 + len(fix_distances_high_scorers) - 1)


def kstest_fixation_distance_scoring_groups():
    # get data
    df = get_fixations_with_scores()
    df = add_scoring_group_information_to_df(df)

    # fixations_df = pd.read_csv('../data/fixations.csv')[
    #     ['subject_id', 'game_difficulty', 'world_number', 'player_x_field', 'player_y_field', 'weighted_fix_distance_euclidean',
    #      'weighted_fix_distance_manhattan']]
    #
    # df = fixations_df.merge(df, on=['subject_id', 'game_difficulty', 'world_number'], how='left')

    # get weighted fixation distances
    df_high_scorers = df[df['scoring_group'] == 'high']
    df_low_scorers = df[df['scoring_group'] == 'low']
    fix_distances_high_scorers = df_high_scorers['weighted_fix_distance_euclidean']
    fix_distances_low_scorers = df_low_scorers['weighted_fix_distance_euclidean']

    print('H0: Same Distributions | H1: Different Distributions for weighted distances of Low Scorers and of High Scorers')

    # perform (Welch's) t-test
    # t test euclidean distances:
    kstest_result = scipy.stats.kstest(fix_distances_low_scorers, fix_distances_high_scorers, alternative='two-sided')
    print('Test in Weighted Euclidean Distances')
    print(kstest_result)

    fix_distances_high_scorers = df_high_scorers['weighted_fix_distance_manhattan']
    fix_distances_low_scorers = df_low_scorers['weighted_fix_distance_manhattan']

    # perform (Welch's) t-test
    # t test manhattan distances:
    kstest_result = scipy.stats.kstest(fix_distances_low_scorers, fix_distances_high_scorers, alternative='two-sided')
    print('Test in Weighted Manhattan Distances')
    print(kstest_result)


def ttest_mean_level_score_high_scorer_low_scorer():
    # get data
    df = pd.read_csv('../data/level_scores.csv').drop_duplicates()
    df = add_scoring_group_information_to_df(df)

    # get weighted fixation distances
    df_high_scorers = df[df['scoring_group'] == 'high']
    df_low_scorers = df[df['scoring_group'] == 'low']
    level_score_high_scorers = df_high_scorers['level_score']
    level_score_low_scorers = df_low_scorers['level_score']

    print('H0: Same Means | H1: Mean Score per Level for Low Scorers is less than for High Scorers')

    # perform t-test
    ttest_result = scipy.stats.ttest_ind(level_score_low_scorers, level_score_high_scorers,
                                         alternative='less')  # use equal_var=False bc of different sample sizes
    print(ttest_result)
    print('dof=', len(level_score_low_scorers) - 1 + len(level_score_high_scorers) - 1)


if __name__ == '__main__':
    ttest_fixation_distance_scoring_groups()
    print_fixation_distances_per_group()
