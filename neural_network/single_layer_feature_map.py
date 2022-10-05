import numpy as np

import config
from analysis.data_utils import assign_position_to_fields

TARGET_POS2DISCRETE = {448: -1, 1216: 0, 2112: 1}


def create_state_from_str(string):
    return np.fromstring(string)


def create_single_layer_feature_map_from_state(state, target_pos=None):
    """ Creates a feature map consisting of layer with the following encodings:
    player: 1, vehicle: 2, lilypad: 3, target_position: 4
    """
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES))

    # encode target 2position
    if target_pos is not None:
        x_target_pos = int(round(target_pos / config.FIELD_WIDTH))
        y_target_pos = config.N_LANES - 1
        feature_map[x_target_pos, y_target_pos] = 4

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    for obj_type, x, y, width in reversed(state):
        x_start, y, width = assign_position_to_fields(x, y, width)

        # correct player width
        if obj_type == 0:
            width = 1

        # correct for partially visible obstacles
        if x_start < 0:
            width = width + x_start
            x_start = 0

        if x_start + width > config.N_FIELDS_PER_LANE:
            width = config.N_FIELDS_PER_LANE - x_start

        feature_map[x_start:x_start + width, y] = obj_type + 1

    feature_map = np.rot90(feature_map)

    # invert y axis
    feature_map = np.flip(feature_map, axis=1)

    return feature_map
