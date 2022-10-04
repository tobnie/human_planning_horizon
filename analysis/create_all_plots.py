from analysis.actions.action_plots import plot_action_distributions, plot_actions_situation_heatmaps
from analysis.data_utils import get_river_data, get_street_data
from analysis.gaze.blink_rates import plot_blink_rate_over_level_score, plot_blink_rate_over_score, plot_blink_rates
from analysis.gaze.blinks import plot_IBI_per_subject
from analysis.gaze.fixations import load_fixations, plot_avg_fixation_distance_per_subject, plot_fixation_angle_per_position, \
    plot_fixation_distance_box_per_region, plot_fixation_distance_per_meta_data, plot_fixation_distance_per_position, plot_fixations_kde, \
    plot_gaze_x_position_relative_to_player, \
    plot_gaze_y_position_relative_to_player, \
    plot_mfd_per_level_score, \
    plot_polar_hist_for_fixations_per_position, plot_weighted_fixations_relative_to_player, \
    plot_weighted_fixations_relative_to_player_per_fixated_object, \
    print_fixations_on_target_for_region
from analysis.performance.experts_vs_novices import print_fixation_distances_per_group
from analysis.performance.performances import histogram_over_avg_trial_times, plot_causes_of_death, plot_last_lanes_lost_games, \
    plot_mean_score_per_level, plot_performance_per_difficulty, \
    print_average_game_endings
from analysis.player.player_position_heatmap import plot_player_position_heatmap, plot_player_position_heatmap_per_target_position
from analysis.player.river_section_entrance import plot_entrance_of_river_section
from analysis.pupil_size.pupil_size import plot_pupil_size, plot_pupil_size_for_each_game


def try_except_plot(func):
    try:
        func()
    except Exception as err:
        print(f'Plot {func} failed with the following error:\n')
        print(err)


if __name__ == '__main__':
    # performance
    try_except_plot(plot_performance_per_difficulty)
    try_except_plot(print_average_game_endings)
    try_except_plot(plot_causes_of_death)  # TODO
    try_except_plot(plot_last_lanes_lost_games)  # TODO lane 0
    try_except_plot(histogram_over_avg_trial_times)
    try_except_plot(plot_mean_score_per_level)

    # experts vs novices
    try_except_plot(print_fixation_distances_per_group)

    # actions
    try_except_plot(plot_action_distributions)
    try_except_plot(plot_actions_situation_heatmaps)

    # player position
    try_except_plot(plot_player_position_heatmap)
    try_except_plot(plot_player_position_heatmap_per_target_position)  # TODO
    try_except_plot(plot_entrance_of_river_section)

    # blinks
    try_except_plot(plot_blink_rate_over_score)
    try_except_plot(plot_blink_rate_over_level_score)
    try_except_plot(plot_blink_rates)

    # ibi
    try_except_plot(plot_IBI_per_subject)

    # pupil size
    try_except_plot(plot_pupil_size)
    try_except_plot(plot_pupil_size_for_each_game)

    # fixations
    try_except_plot(plot_avg_fixation_distance_per_subject)

    try_except_plot(plot_fixations_kde)

    try_except_plot(plot_polar_hist_for_fixations_per_position)

    try_except_plot(plot_avg_fixation_distance_per_subject)
    try_except_plot( plot_mfd_per_level_score)

    df = load_fixations()
    plot_fixation_angle_per_position(df)
    plot_fixation_distance_per_position(df)
    plot_fixation_distance_box_per_region(df)

    try_except_plot(plot_gaze_y_position_relative_to_player)
    try_except_plot(print_fixations_on_target_for_region)
    try_except_plot(plot_weighted_fixations_relative_to_player_per_fixated_object)  # TODO run
    try_except_plot(plot_weighted_fixations_relative_to_player)
    try_except_plot(plot_gaze_x_position_relative_to_player)

    plot_fixation_distance_per_meta_data(df)

    df_street = get_street_data(df)
    plot_fixation_distance_per_meta_data(df_street, subfolder='street/')

    df_river = get_river_data(df)
    plot_fixation_distance_per_meta_data(df_street, subfolder='river/')
