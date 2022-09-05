import numpy as np

import config
from analysis.data_utils import create_state_from_string, read_subject_data
from analysis.world.feature_maps import get_area_around_player
from data.preprocessing import create_feature_map_from_state


def invert_water_in_fm(fm):
    fm[8:14, :, 2] = 1 - fm[8:14, :, 2]
    return fm


def get_avoidance_map_from_state(state, player_x, player_y, radius=1):
    fm = create_feature_map_from_state(state)

    # invert water
    fm = invert_water_in_fm(fm)

    fm_around_player = get_area_around_player(fm, player_x, player_y, radius=radius)
    avoidance_map = fm_around_player.sum(axis=-1)

    avoidance_map[avoidance_map >= 1] = 1

    return avoidance_map


# df = read_data()
df = read_subject_data('AN06AN')
player_width = config.PLAYER_WIDTH
field_width = config.FIELD_WIDTH
df_reduced = df[['state', 'player_x_field', 'player_y_field']]
test = df_reduced['state']
avoidance_maps = df_reduced.apply(
    lambda x: get_avoidance_map_from_state(create_state_from_string(x['state']), x['player_x_field'], x['player_y_field']), axis=1)
avoidance_map_arr = np.array(avoidance_maps)
# TODO make correct array out of it
print(avoidance_map_arr.shape)
print(avoidance_map_arr)
