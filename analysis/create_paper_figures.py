import pandas as pd
import seaborn as sns

from analysis.gaze.fixations import plot_fixation_distance_box_per_region, plot_fixation_distance_per_position, \
    plot_avg_fixation_distance_per_subject
from analysis.performance.experts_vs_novices import plot_fixation_distance_box_scoring_groups, print_fixation_distances_per_group, \
    plot_fixation_distance_scoring_groups_for_different_splits
from analysis.performance.performances import plot_mean_score_per_level, histogram_over_avg_trial_times

if __name__ == '__main__':
    # plot_fixation_distance_scoring_groups_for_different_splits()

    plot_avg_fixation_distance_per_subject()
    histogram_over_avg_trial_times()
    #
    sns.set_style("whitegrid")
    plot_mean_score_per_level()
    plot_fixation_distance_box_scoring_groups()
    #
    df = pd.read_csv('../data/fixations.csv')
    plot_fixation_distance_box_per_region(df)
    plot_fixation_distance_per_position(df)
    print_fixation_distances_per_group()

