import os

from analysis.data_utils import read_subject_data
from analysis.gaze.adaptiveIDT.data_filtering import PIX2DEG

input_file = 'remodnav_input.csv'
output_file = 'remodnav_output.tsv'
sampling_rate = 60
MIN_SACCADE_DURATION = 20 * 1e-3
savgol_length = 3 * MIN_SACCADE_DURATION

df = read_subject_data('ED06RA')
trial_df = df[df['trial'] == 35]
x, y = trial_df['gaze_x'].to_numpy(), trial_df['gaze_y'].to_numpy()
trial_df[['gaze_x', 'gaze_y']].to_csv(input_file, sep='\t', index=False, header=False)

command = 'remodnav {} {} {} {} --min-saccade-duration {} --savgol-length {}'.format(input_file, output_file, PIX2DEG, sampling_rate,
                                                                                     MIN_SACCADE_DURATION, savgol_length)
os.system(command)
