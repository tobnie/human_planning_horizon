from analysis.gaze.fixations import ttest_fixation_distance_street_river, kstest_fixation_distance_street_river
from analysis.performance.experts_vs_novices import ttest_fixation_distance_scoring_groups, kstest_fixation_distance_scoring_groups

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
