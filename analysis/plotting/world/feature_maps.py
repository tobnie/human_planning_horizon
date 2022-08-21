import numpy as np

import config
from analysis.analysis_utils import assign_position_to_fields, OBJECT_TO_INT


def create_feature_map_from_state(state):
    # TODO not working correctly yet, it seems?

    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES, 3))

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    for obj_type, x, y, width in state:
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

        feature_map[x_start:x_start + width, y, obj_type] = 1

    feature_map = np.rot90(feature_map)

    # invert y axis
    feature_map = np.flip(feature_map, axis=1)

    return feature_map


def states_to_feature_maps(list_of_states):
    """ Transforms a list of states into an array of feature maps. States are distributed along axis 0.
     Feature Maps have the following form: ['state', 'x', 'y', 'type']
     Types are Player: 0, Vehicle: 1, LilyPad: 2"""
    return np.array([create_feature_map_from_state(state) for state in list_of_states])


def invert_feature_map(feature_map):
    return 1 - feature_map


def get_feature_map_for_object(feature_map, obj_name):
    """ Returns the feature map for the given object type.
    obj_name must either be 'player', 'vehicle', or 'lilypad'"""
    obj_type = OBJECT_TO_INT[obj_name.lower()]
    if feature_map.ndim == 3:
        return feature_map[:, :, obj_type]
    else:
        return feature_map[:, :, :, obj_type]


def get_feature_map_for_player(feature_map):
    return get_feature_map_for_object(feature_map, 'player')


def get_feature_map_distribution_for_object(feature_maps, obj_name):
    """ Returns the distribution of the given object type in the feature map.
    obj_name must either be 'player', 'vehicle', or 'lilypad'"""
    fm = get_feature_map_for_object(feature_maps, obj_name)
    return fm.sum(axis=0)


def get_feature_map_distribution_for_water(feature_maps):
    """ Returns the distribution of the water in the feature map."""
    lilypad_fm = get_feature_map_for_object(feature_maps, 'lilypad')

    water_fm = np.zeros_like(lilypad_fm)
    water_fm[:, 8:14] = invert_feature_map(lilypad_fm[:, 8:14])

    return water_fm.sum(axis=0)


def get_feature_map_distribution_for_player(feature_maps):
    return get_feature_map_distribution_for_object(feature_maps, 'player')


def get_feature_map_for_avoidance(feature_maps):
    """ Returns the feature map for the objects that should be avoided in the feature map."""
    # TODO need to distinguish between street and water section, since in street section the player is standing still, while it is moving in the river section

    # get feature maps
    vehicle_fm = get_feature_map_distribution_for_object(feature_maps, 'vehicle')
    water_fm = get_feature_map_distribution_for_water(feature_maps)

    # combine feature maps
    avoidance_fm = vehicle_fm + water_fm
    return avoidance_fm


def get_feature_map_distribution_for_avoidance(feature_maps):
    """ Returns the distribution of the objects that should be avoided in the feature map."""
    avoidance_fm = get_feature_map_for_avoidance(feature_maps)
    return avoidance_fm


def get_avoidance_distribution_around_player_from_state_dict(state_dict, radius=1):
    return {key: get_avoidance_distribution_around_player_from_state_list(states, radius=radius) for key, states in state_dict.items()}


def get_avoidance_distribution_around_player_from_state_list(states, radius=1):
    fms = states_to_feature_maps(states)
    fms_around_player = get_area_around_player(fms, radius=radius)
    return get_feature_map_distribution_for_avoidance(fms_around_player)


def get_player_position_in_map(feature_map):
    """ Returns the position of the player in the feature map as (x, y)-tuple."""
    player_fm = get_feature_map_for_player(feature_map)
    indices = np.where(player_fm == 1)
    return indices[0][0], indices[1][0]


def get_area_around_field(feature_map, x, y, radius=1, pad_sides=True):
    """ Returns the area around a given field in the feature map (preserves depth)"""

    if pad_sides:
        # pad sides with ones
        padded_fm = np.pad(feature_map, ((radius, radius), (radius, radius), (0, 0)), 'constant', constant_values=1)
        feature_map = padded_fm
        # shift coordinates of desired field to account for padding
        x = x + radius
        y = y + radius
    area_around_field = feature_map[x - radius:x + radius + 1, y - radius:y + radius + 1]

    return area_around_field


def get_area_around_player(feature_map, radius=1):
    """ Returns the area around the player in the feature map (preserves depth)"""
    player_x, player_y = get_player_position_in_map(feature_map)

    if feature_map.ndim == 4:
        return get_area_around_player_list(feature_map, radius)

    area_around_player = get_area_around_field(feature_map, player_x, player_y, radius)
    return area_around_player


def get_area_around_player_list(feature_maps, radius=1):
    """ Returns the area around the player in the feature map for each state (preserves depth)"""
    fms_around_player = np.zeros((feature_maps.shape[0], radius * 2 + 1, radius * 2 + 1, feature_maps.shape[3]))

    for i, fm in enumerate(feature_maps):
        fms_around_player[i] = get_area_around_player(fm, radius)

    return fms_around_player
