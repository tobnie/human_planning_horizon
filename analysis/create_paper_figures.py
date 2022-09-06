import pandas as pd
import seaborn as sns

from analysis.gaze.fixations import plot_fixation_distance_box_per_region, plot_fixation_distance_per_position
from analysis.performance.experts_vs_novices import plot_fixation_distance_box_scoring_groups, print_fixation_distances_per_group
from analysis.performance.performances import plot_mean_score_per_level, histogram_over_avg_trial_times

if __name__ == '__main__':
    histogram_over_avg_trial_times()

    sns.set_style("whitegrid")
    plot_mean_score_per_level()
    plot_fixation_distance_box_scoring_groups()

    # #
    df = pd.read_csv('fixations.csv')
    plot_fixation_distance_box_per_region(df)
    plot_fixation_distance_per_position(df)
    print_fixation_distances_per_group()

