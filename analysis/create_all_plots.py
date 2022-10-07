import sys
import traceback
import seaborn as sns

sns.set_context(rc={'patch.linewidth': 0.7})

from analysis.gaze.blink_rates import plot_blink_rate_over_level_score, plot_blink_rate_over_score, plot_blink_rates
from analysis.gaze.fixations import plot_MFD_diff_river_street_over_score, plot_fixation_KDE_relative_to_player, \
    plot_fixation_heatmap, plot_mfd_heatmap, plot_mfd_per_level_score, plot_mfd_per_region, plot_mfd_per_score, \
    print_fixations_on_target_for_region
from analysis.performance.experts_vs_novices import print_fixation_distances_per_group
from analysis.performance.performances import histogram_over_avg_trial_times, plot_last_lanes_lost_games, \
    plot_mean_score_per_level, plot_performance_per_difficulty, \
    print_average_game_endings
from analysis.player.player_position_heatmap import plot_player_position_heatmap, plot_player_position_heatmap_per_target_position
from analysis.player.river_section_entrance import plot_entrance_of_river_section
from analysis.pupil_size.pupil_size import plot_pupil_size
from analysis.run_statistical_tests import run_tests


def try_except_plot(func):
    try:
        func()
    except Exception:
        print(f'Plot {func} failed with the following error:\n')
        print(traceback.format_exc())


if __name__ == '__main__':
    plot_mfd_heatmap()

    try_except_plot(plot_performance_per_difficulty)

    try_except_plot(plot_fixation_heatmap)

    try_except_plot(plot_fixation_KDE_relative_to_player)

    try_except_plot(plot_mfd_per_region)

    # redirect prints to file:
    orig_stdout = sys.stdout
    f = open('../thesis/1descriptive/1performance/performances.txt', 'w+')
    sys.stdout = f
    try_except_plot(print_average_game_endings)
    sys.stdout = orig_stdout
    f.close()

    try_except_plot(plot_last_lanes_lost_games)  # TODO lane 0 + 7
    f = open('../thesis/1descriptive/1performance/trial_times.txt', 'w+')
    sys.stdout = f
    try_except_plot(histogram_over_avg_trial_times)
    sys.stdout = orig_stdout
    f.close()

    f = open('../thesis/1descriptive/1performance/score_per_level.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_mean_score_per_level)
    sys.stdout = orig_stdout
    f.close()

    # experts vs novices
    f = open('../thesis/3experts_vs_novices/mfd_per_group.txt', 'w+')
    sys.stdout = f
    try_except_plot(print_fixation_distances_per_group)
    sys.stdout = orig_stdout
    f.close()

    try_except_plot(plot_mfd_per_level_score)

    # player position
    try_except_plot(plot_player_position_heatmap)
    try_except_plot(plot_player_position_heatmap_per_target_position)

    try_except_plot(plot_entrance_of_river_section)

    # blinks
    f = open('../thesis/3experts_vs_novices/blink_rate_over_score_lin_reg.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_blink_rate_over_score)
    sys.stdout = orig_stdout
    f.close()

    f = open('../thesis/3experts_vs_novices/blink_rate_over_level_score_lin_reg.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_blink_rate_over_level_score)
    sys.stdout = orig_stdout
    f.close()

    f = open('../thesis/3experts_vs_novices/blink_rate_per_score.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_blink_rates)
    sys.stdout = orig_stdout
    f.close()

    # pupil size
    f = open('../thesis/2river_vs_street/pupil_size.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_pupil_size)
    sys.stdout = orig_stdout
    f.close()

    # fixations
    f = open('../thesis/3experts_vs_novices/mfd_per_level_score_lin_reg.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_mfd_per_score)
    sys.stdout = orig_stdout
    f.close()

    f = open('../thesis/3experts_vs_novices/mfd_diff_per_score_lin_reg.txt', 'w+')
    sys.stdout = f
    try_except_plot(plot_MFD_diff_river_street_over_score)
    sys.stdout = orig_stdout
    f.close()

    try_except_plot(plot_mfd_heatmap)

    f = open('../thesis/2river_vs_street/fixations_on_target.txt', 'w+')
    sys.stdout = f
    try_except_plot(print_fixations_on_target_for_region)  # TODO maybe also make bar plot / maybe was aus KDE rausrechnen?
    sys.stdout = orig_stdout
    f.close()

    # other statistical tests
    f = open('../thesis/other_tests.txt', 'w+')
    sys.stdout = f
    run_tests()
    sys.stdout = orig_stdout
    f.close()
