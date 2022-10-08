import numpy as np

import config
from analysis.data_utils import assign_object_position_to_fields_street, assign_object_position_to_fields_water, \
    assign_player_position_to_field

TARGET_POS2DISCRETE = {448: -1, 1216: 0, 2112: 1}


def create_state_from_str(string):
    return np.fromstring(string)


def create_single_layer_feature_map_from_state(state, target_pos=None):
    """ Creates a feature map consisting of layer with the following encodings:
    player: 1, vehicle: 2, lilypad: 3, target_position: 4
    """
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES))

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    _, player_x_start, _, _ = state[0]
    player_x_center = player_x_start + config.PLAYER_WIDTH / 2

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    for obj_type, x, y, width in reversed(state):
        if obj_type == 0:
            x_start, y = assign_player_position_to_field(x, y)
            x_end = x_start + 1
        elif obj_type == 1:
            x_start, x_end, y = assign_object_position_to_fields_street(x, y, width, player_x_start)
        else:
            x_start, x_end, y = assign_object_position_to_fields_water(x, y, width, player_x_center)

        feature_map[x_start:x_end, y] = obj_type + 1

    # encode target position
    if target_pos is not None:
        x_target_pos = int(target_pos // config.FIELD_WIDTH)
        y_target_pos = config.N_LANES - 1
        feature_map[x_target_pos, y_target_pos] = 4

    feature_map = np.rot90(feature_map)

    # invert y axis
    feature_map = np.flip(feature_map, axis=1)

    return feature_map
