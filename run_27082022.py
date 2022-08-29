from analysis.plotting.gaze.gaze_per_position import run_gaze_per_position_plots
from neural_network.single_layer_feature_map import run_create_IO_data_for_NN

# print('\n\n --------------------------')
# print('Starting to create NN Data...')
# try:
#     run_create_IO_data_for_NN()
# except Exception as e:
#     print('Creating Input Output data for nn failed :(')
#     print('\n Error message: \n' + str(e))

# TODO individually for: NI07LU, ZI01SU, MA03CL, KR07HA
# TODO run code below again
print('\n\n --------------------------')
print('Starting to create plots...')
run_gaze_per_position_plots()

