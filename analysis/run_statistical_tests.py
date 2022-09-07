from analysis.gaze.fixations import ttest_fixation_distance_street_river, kstest_fixation_distance_street_river
from analysis.performance.experts_vs_novices import ttest_fixation_distance_scoring_groups, kstest_fixation_distance_scoring_groups
from analysis.performance.performances import anova_mean_level_score, ttest_mean_level_score_easy_normal, ttest_mean_level_score_normal_hard

if __name__ == '__main__':
    print('\n---- t-test for fixation distances between different scoring groups ----')
    ttest_fixation_distance_scoring_groups()
    print('\n---- Kolmogorov Smirnov for fixation distances between different scoring groups ----')
    kstest_fixation_distance_scoring_groups()

    print('\n\n')

    print('\n---- t-test for fixation distances river / street ----')
    ttest_fixation_distance_street_river()
    print('\n---- Kolmogorov Smirnov for fixation distances river / street ----')
    kstest_fixation_distance_street_river()

    print('\n---- ANOVA for mean score per level ----')
    anova_mean_level_score()

    print('\n---- t-test for mean score per level easy / normal ----')
    ttest_mean_level_score_easy_normal()
    print('\n---- t-test for mean score per level normal / hard ----')
    ttest_mean_level_score_normal_hard()
