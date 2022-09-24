from analysis.gaze.blink_rates import save_blink_rates
from analysis.gaze.blinks import save_blinks_and_IBIs
from analysis.gaze.fixations import save_fixations
from analysis.gaze.saccades import save_saccades
from analysis.performance.performances import save_performance_stats, save_game_durations, save_level_scores
from data.preprocessing import run_preprocessing

save_fixations()

# run_preprocessing()
#
# # create separate dataframes for faster loading
# save_blinks_and_IBIs()
# save_blink_rates()
# save_fixations()
# save_saccades()
# save_performance_stats()
# save_game_durations()
# save_level_scores()
