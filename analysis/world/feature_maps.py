import numpy as np

from analysis.analysis_utils import OBJECT_TO_INT, states_to_feature_maps


def invert_feature_map(feature_map):
    return 1 - feature_map


def get_feature_map_for_object(feature_map, obj_name):
    """ Returns the feature map for the given object type.
    obj_name must either be 'player', 'vehicle', or 'lilypad'"""
    obj_type = OBJECT_TO_INT[obj_name.lower()]
    return feature_map[:, :, obj_type]


def get_feature_map_for_player(feature_map):
    return get_feature_map_for_object(feature_map, 'player')


def get_feature_map_distribution_for_object(feature_map, obj_name):
    """ Returns the distribution of the given object type in the feature map.
    obj_name must either be 'player', 'vehicle', or 'lilypad'"""
    fm = get_feature_map_for_object(feature_map, obj_name)
    return fm.sum(axis=0)


def get_feature_map_distribution_for_water(feature_maps):
    """ Returns the distribution of the water in the feature map."""
    lilypad_fm = get_feature_map_for_object(feature_maps, 'lilypad')

    water_fm = np.zeros_like(lilypad_fm)
    water_fm[:, 8:14] = invert_feature_map(lilypad_fm[:, 8:14])

    return water_fm.sum(axis=0)


def get_feature_map_distribution_for_player(feature_map):
    return get_feature_map_distribution_for_object(feature_map, 'player')


def get_avoidance_map(feature_map):
    """ Returns the feature map for the objects that should be avoided in the feature map."""
    # TODO need to distinguish between street and water section, since in street section the player is standing still, while it is moving in the river section

    # get feature maps
    vehicle_fm = get_feature_map_distribution_for_object(feature_map, 'vehicle')
    water_fm = get_feature_map_distribution_for_water(feature_map)

    # combine feature maps
    avoidance_fm = vehicle_fm + water_fm
    return avoidance_fm


def get_feature_map_distribution_for_avoidance(feature_map):
    """ Returns the distribution of the objects that should be avoided in the feature map."""
    avoidance_fm = get_avoidance_map(feature_map)
    return avoidance_fm


def get_avoidance_distribution_around_player_from_state_dict(state_dict, radius=1):
    return {key: get_avoidance_distribution_around_player_from_state_list(states, radius=radius) for key, states in state_dict.items()}


def get_avoidance_distribution_around_player_from_state_list(states, radius=1):
    fms = states_to_feature_maps(states)
    fms_around_player = get_area_around_player(fms, None, None, radius=radius)  # TODO
    avoidance_map = get_feature_map_distribution_for_avoidance(fms_around_player)
    return avoidance_map


def get_player_position_in_map(feature_map):
    """ Returns the position of the player in the feature map as (x, y)-tuple."""
    player_fm = get_feature_map_for_player(feature_map)
    indices = np.where(player_fm == 1)
    return indices[0][0], indices[1][0]


def get_area_around_field(feature_map, x, y, radius=1, pad_sides=True):
    """ Returns the area around a given field in the feature map (preserves depth)"""
    x = int(x)
    y = int(y)
    if pad_sides:
        # pad sides with ones
        padded_fm = np.pad(feature_map, ((radius, radius), (radius, radius), (0, 0)), 'constant', constant_values=1)
        feature_map = padded_fm
        # shift coordinates of desired field to account for padding
        x = x + radius
        y = y + radius

    area_around_field = feature_map[y - radius:y + radius + 1, x - radius:x + radius + 1]

    return area_around_field


def get_area_around_player(feature_map, player_x, player_y, radius=1):
    """ Returns the area around the player in the feature map (preserves depth)"""
    area_around_player = get_area_around_field(feature_map, player_x, player_y, radius)
    return area_around_player


def get_area_around_player_list(feature_maps, radius=1):
    """ Returns the area around the player in the feature map for each state (preserves depth)"""
    fms_around_player = np.zeros((feature_maps.shape[0], radius * 2 + 1, radius * 2 + 1, feature_maps.shape[3]))

    for i, fm in enumerate(feature_maps):
        fms_around_player[i] = get_area_around_player(fm, radius)

    return fms_around_player


def filter_times_actions_fms_for_player_position(times_actions_fms, row):
    """ Returns only time-action-fm-pairs where the player is in the given row"""
    filtered_indices = []
    for i, (time, action, fm) in enumerate(times_actions_fms):
        if fm[row, :, 0].sum() > 0:
            filtered_indices.append(i)

    filtered_array = [times_actions_fms[i] for i in filtered_indices]
    return filtered_array


def filter_fms_for_player_position(fms, row):
    """ Returns only feature maps where the player is in the given row"""
    filter_array = np.zeros(fms.shape[0])
    for i, fm in enumerate(fms):
        if fm[row, :, 0].sum() > 0:
            filter_array[i] = 1

    filtered_fms = fms[filter_array == 1]
    return filtered_fms
