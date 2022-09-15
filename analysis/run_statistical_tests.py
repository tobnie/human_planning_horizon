from analysis.gaze.blink_rates import ttest_blink_rate_street_river, kstest_blink_rate_distance_street_river
from analysis.gaze.fixations import ttest_fixation_distance_street_river, kstest_fixation_distance_street_river
from analysis.performance.experts_vs_novices import ttest_fixation_distance_scoring_groups, kstest_fixation_distance_scoring_groups, \
    ttest_mean_level_score_high_scorer_low_scorer
from analysis.performance.performances import anova_mean_level_score, ttest_mean_time_easy_normal, ttest_mean_time_normal_hard

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

    print('\n\n')

    print('\n---- t-test for blink rates river / street ----')
    ttest_blink_rate_street_river()
    print('\n---- Kolmogorov Smirnov for blink rates river / street ----')
    kstest_blink_rate_distance_street_river()

    print('\n\n')

    print('\n---- ANOVA for mean score per level ----')
    anova_mean_level_score()

    print('\n\n')

    print('\n---- t-test for mean time per level easy / normal ----')
    ttest_mean_time_easy_normal()
    print('\n---- t-test for mean time per level normal / hard ----')
    ttest_mean_time_normal_hard()

    print('\n\n')

    print('\n---- t-test for mean score per level high-scorer / low-scorer ----')
    ttest_mean_level_score_high_scorer_low_scorer()
