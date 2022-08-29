import json

import pandas as pd


def load_world_info(difficulty, world_number) -> None:
    """ Loads lanes from a json file with the specified world name. """
    # load json at given path
    with open(f'../game/levels/{difficulty}/world_{world_number}.json', 'r', encoding='utf-8') as f:
        world_dict = json.load(f)
    return world_dict


def get_starting_lane_direction_from_world_info(world_info):
    # get last directed lane
    first_directed_lane = world_info['lanes'][-2]
    direction_of_first_lane = first_directed_lane['direction']
    direction_of_first_lane = 'left' if direction_of_first_lane == -1 else 'right'
    return direction_of_first_lane


def get_left_or_right_world_df():
    diffs = []
    world_numbers = []
    left_or_right_starting_lane = []

    for difficulty in ['easy', 'normal', 'hard']:
        for i in range(20):
            world_dict = load_world_info(difficulty, i)
            direction_of_first_lane = get_starting_lane_direction_from_world_info(world_dict)

            left_or_right_starting_lane.append(direction_of_first_lane)
            diffs.append(difficulty)
            world_numbers.append(i)

    df = pd.DataFrame.from_dict({
        'game_difficulty': diffs,
        'world_number': world_numbers,
        'first_direction': left_or_right_starting_lane
    })

    return df


def add_world_direction_info_to_df(df):
    direction_df = get_left_or_right_world_df()
    df = df.merge(direction_df, on=['game_difficulty', 'world_number'], how='left')
    return df

# TODO test and transformations for state