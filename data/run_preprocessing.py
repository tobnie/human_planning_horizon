
from analysis.gaze.blink_rates import save_blink_rates
from analysis.gaze.blinks import save_blinks_and_IBIs
from analysis.gaze.events.run_remodnav import save_remodnav_fixations
from analysis.gaze.fixations import save_fixations
from analysis.gaze.saccades import save_saccades
from analysis.performance.performances import save_game_durations, save_level_scores, save_performance_stats
from analysis.world.feature_maps import save_states_with_identifiers
from data.preprocessing import run_preprocessing, save_world_direction_types

run_preprocessing()

# create separate dataframes for faster loading
save_blinks_and_IBIs()
save_blink_rates()
save_fixations()
save_remodnav_fixations()
save_saccades()
save_performance_stats()
save_game_durations()
save_level_scores()

save_states_with_identifiers()

# the following line only needed to be run once since it should not change
# save_world_direction_types()

# TODO rerun fixation plots (AND ALL OTHER PLOTS AND TESTS!) after preprocessing
