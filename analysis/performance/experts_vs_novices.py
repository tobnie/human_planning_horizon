import pandas as pd
import seaborn as sns
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl

from analysis import paper_colors


def get_fixations_with_scores():
    return pd.read_csv('fixations.csv')


def add_scoring_group_information_to_df(df):
    performance_stats = pd.read_csv('performance_stats.csv', index_col=0)
    median_score = performance_stats['score'].median()

    # classify based on score
    df['scoring_group'] = df['score'].apply(lambda x: 'low' if x < median_score else 'high')
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

    edge_colors = [paper_colors.C0, paper_colors.C1]
    box_colors = [paper_colors.C0_soft, paper_colors.C1_soft]

    # create boxplot
    sns.set_style("whitegrid")
    ax = sns.boxplot(data=df, x='scoring_group', y='weighted_fix_distance_manhattan', width=0.2, linewidth=1.5,
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
    ttest_result = scipy.stats.ttest_ind(fix_distances_low_scorers, fix_distances_high_scorers, equal_var=False,
                                         alternative='less')  # use equal_var=False bc of different sample sizes
    print('Test in Weighted Euclidean Distances')
    print(ttest_result)

    fix_distances_high_scorers = df_high_scorers['weighted_fix_distance_manhattan']
    fix_distances_low_scorers = df_low_scorers['weighted_fix_distance_manhattan']

    # perform (Welch's) t-test
    # t test manhattan distances:
    ttest_result = scipy.stats.ttest_ind(fix_distances_low_scorers, fix_distances_high_scorers, equal_var=False,
                                         alternative='less')  # use equal_var=False bc of different sample sizes
    print('Test in Weighted Manhattan Distances')
    print(ttest_result)


def kstest_fixation_distance_scoring_groups():
    # get data
    df = get_fixations_with_scores()
    df = add_scoring_group_information_to_df(df)

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


if __name__ == '__main__':
    ttest_fixation_distance_scoring_groups()
    print_fixation_distances_per_group()
