import numpy as np
from matplotlib import pyplot as plt

import config
from analysis.data_utils import assign_object_position_to_fields, assign_object_position_to_fields_street, \
    assign_object_position_to_fields_water, \
    assign_player_position_to_field, create_state_from_string, read_data
from analysis.world.world_coordinates import plot_state


def invert_feature_map(feature_map):
    return 1 - feature_map


def get_feature_map_for_player(feature_map):
    return feature_map[:, :, 0]


def get_avoidance_map_street(feature_map):
    return feature_map[:, :, 1]


def get_avoidance_map_water(feature_map):
    """ Returns the distribution of the water in the feature map."""
    lilypad_fm = feature_map[:, :, 2]

    water_fm = np.zeros_like(lilypad_fm)
    water_fm[1:7] = invert_feature_map(lilypad_fm[1:7])

    return water_fm


def get_avoidance_map(feature_map):
    """ Returns the feature map for the objects that should be avoided in the feature map."""
    # TODO need to distinguish between street and water section, since in street section the player is standing still, while it is moving in the river section?

    # get feature maps
    vehicle_fm = get_avoidance_map_street(feature_map)
    water_fm = get_avoidance_map_water(feature_map)

    # combine feature maps
    avoidance_fm = vehicle_fm + water_fm
    return avoidance_fm


def get_feature_map_distribution_for_avoidance(feature_map):
    """ Returns the distribution of the objects that should be avoided in the feature map."""
    avoidance_fm = get_avoidance_map(feature_map)
    return avoidance_fm


def get_player_position_in_map(feature_map):
    """ Returns the 2position of the player in the feature map as (x, y)-tuple."""
    player_fm = get_feature_map_for_player(feature_map)
    indices = np.where(player_fm == 1)
    return indices[0][0], indices[1][0]


def get_area_around_field(feature_map, y, x, radius=1, pad_sides=True):
    """ Returns the area around a given field in the feature map (preserves depth)"""
    x = int(x)
    y = int(y)
    if pad_sides:
        # pad sides with ones
        padded_fm = np.pad(feature_map, ((radius, radius), (radius, radius)), 'constant', constant_values=1)
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


def classify_situation(situation):
    """ Adds a unique identifier to the given situation. A situation is a cutout from an avoidance map around the player."""
    # flatten array for turning into string
    flattened_array = situation.flatten()

    situation_identifier = flattened_array.astype(int).astype(str)

    # concat to string
    situation_identifier = ''.join(situation_identifier)

    return situation_identifier


def convert_state_to_fm_and_classify(state_from_df, radius=1):
    state = create_state_from_string(state_from_df)
    state_fm = create_feature_map_from_state(state)
    avoidance_map = get_avoidance_map(state_fm)
    player_x, player_y = get_player_position_in_map(state_fm)
    situation = get_area_around_player(avoidance_map, player_x, player_y, radius)
    situation_identifier = classify_situation(situation)
    return situation_identifier


def get_avoidance_map_from_state_identifier(identifier):
    n = int(np.sqrt(len(identifier)))
    avoidance_map = np.fromstring(identifier, dtype='u1') - ord('0')
    avoidance_map = avoidance_map.reshape((n, n))
    return avoidance_map


def save_states_with_identifiers():
    df = read_data()
    df = df[['subject_id', 'game_difficulty', 'world_number', 'time', 'action', 'state', 'player_x_field', 'player_y_field', 'region',
             'lane_type']]

    df['state_identifier'] = df['state'].apply(lambda x: convert_state_to_fm_and_classify(x, radius=3))

    df.drop(columns=['state'], inplace=True)
    df.to_csv('../data/situations.csv')


def create_feature_map_from_state(state):
    feature_map = np.zeros((config.N_FIELDS_PER_LANE, config.N_LANES, 3))

    # object types are Player: 0, Vehicle: 1, LilyPad: 2
    _, player_x_start, _, _ = state[0]
    player_x_center = player_x_start + config.PLAYER_WIDTH / 2
    for obj_type, x, y, width in state:

        if obj_type == 0:
            x_start, y = assign_player_position_to_field(x, y)
            x_end = x_start + 1
        elif obj_type == 1:
            x_start, x_end, y = assign_object_position_to_fields_street(x, y, width, player_x_start)
        else:
            x_start, x_end, y = assign_object_position_to_fields_water(x, y, width, player_x_center)

        feature_map[x_start:x_end, y, obj_type] = 1

    feature_map = np.rot90(feature_map)

    return feature_map
