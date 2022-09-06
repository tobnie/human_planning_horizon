from analysis.plotting.actions.action_plots import plot_and_save_action_distributions
from analysis.plotting.gaze.events.blink_plot import plot_and_save_blinks
from analysis.plotting.gaze.events.saccade_plot import plot_and_save_saccades
from analysis.gaze.gaze_plot import create_gaze_and_path_plots
from analysis.plotting.gaze.polar_plot import plot_gaze_angles_for_all_subjects
from analysis.plotting.world.avoidance_plots import plot_and_save_avoidance_maps

# print('Creating gaze plots for all...')
# create_and_save_gaze_density_for_all()
# print('Creating gaze plots per row...')
# create_and_save_gaze_densities_per_row()
# print('Creating Action Distributions...')
# plot_and_save_action_distributions()
# print('Creating gaze angles...')
# plot_gaze_angles_for_all_subjects()
# print('Creating Avoidance Maps...')
# plot_and_save_avoidance_maps()
# print('Creating Blink Plots...')
# plot_and_save_blinks()
# print('Creating Fixation Plots...')
# plot_and_save_fixations()
# print('Creating Fixation Times Plots...')
# plot_and_save_fixation_times()
# print('Creating Saccade Plots...')
# plot_and_save_saccades()
print('Creating gaze and path plots...')
create_gaze_and_path_plots()
