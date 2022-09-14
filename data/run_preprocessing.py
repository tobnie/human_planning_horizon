from analysis.gaze.fixations import save_fixation_info
from analysis.gaze.saccades import save_saccades
from analysis.performance.performances import save_performance_stats, save_game_durations, save_level_scores
from data.preprocessing import run_preprocessing

# run_preprocessing()

# create separate dataframes for faster loading
save_fixation_info()
# save_saccades()
# save_performance_stats()
# save_game_durations()
# save_level_scores()
