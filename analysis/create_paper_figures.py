import seaborn as sns

from analysis.gaze.fixations import plot_mfd_per_region, plot_mfd_heatmap, plot_mfd_per_score
from analysis.performance.experts_vs_novices import plot_fixation_distance_box_scoring_groups, print_fixation_distances_per_group
from analysis.performance.performances import plot_mean_score_per_level, histogram_over_avg_trial_times

if __name__ == '__main__':
    plot_mfd_per_score()
    histogram_over_avg_trial_times()
    #
    sns.set_style("whitegrid")
    plot_mean_score_per_level()
    plot_fixation_distance_box_scoring_groups()
    #
    plot_mfd_per_region()
    plot_mfd_heatmap()
    print_fixation_distances_per_group()
