import numpy as np

from analysis.data_utils import read_data, get_only_onscreen_data

df = read_data()
df = df[['player_x_field', 'player_y_field', 'gaze_x', 'gaze_y']]
df.dropna(inplace=True)

df = get_only_onscreen_data(df)

# get data in arrays
player_pos = df[['player_x_field', 'player_y_field']].to_numpy()

# save data
np.savez_compressed('data/input_position.npz', player_pos)
