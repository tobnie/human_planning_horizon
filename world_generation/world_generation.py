from enum import Enum
from collections import namedtuple

import numpy as np
import pygame

pygame.init()

from world.lane import LaneDirection, StartLane, FinishLane, Lane, StreetLane, WaterLane
import world_generation_utils
import config

ValueWithProbability = namedtuple('ValueWithProbability', ['value', 'probability'])

TARGET_POSITIONS = [2, 10, 18]


class LaneVelocity(Enum):
    """
    Enum for lane velocities.
    """
    SLOW = ValueWithProbability(1, 0.6)
    MEDIUM = ValueWithProbability(2, 0.3)
    FAST = ValueWithProbability(3, 0.1)


class VehicleWidth(Enum):
    """
    Enum for vehicle widths.
    """
    SMALL = ValueWithProbability(1, 0.7)
    MEDIUM = ValueWithProbability(2, 0.2)
    LARGE = ValueWithProbability(3, 0.1)


class LilyPadWidth(Enum):
    """
    Enum for lilypad widths.
    """
    SMALL = ValueWithProbability(1, 0.05)
    MEDIUM = ValueWithProbability(2, 0.35)
    LARGE = ValueWithProbability(3, 0.4)
    XLarge = ValueWithProbability(4, 0.2)


class TargetPositionX(Enum):
    """
    Enum for target position X.
    """
    LEFT = ValueWithProbability(0, 1 / 3)
    MIDDLE = ValueWithProbability(1, 1 / 3)
    RIGHT = ValueWithProbability(2, 1 / 3)


class DistanceBetweenObstacles(Enum):
    """
    Enum for distance between obstacles.
    """
    # TODO even allow SMALL = 1 Field distances?
    SMALL = ValueWithProbability(1, 0.1)
    MEDIUM = ValueWithProbability(2, 0.4)
    LARGE = ValueWithProbability(3, 0.4)
    XLarge = ValueWithProbability(4, 0.1)


class ObstaclesWithoutGap(Enum):
    """
    Enum for number of obstacles without gap.
    """
    SMALL = ValueWithProbability(2, 0.35)
    MEDIUM = ValueWithProbability(3, 0.4)
    LARGE = ValueWithProbability(4, 0.2)
    XLarge = ValueWithProbability(5, 0.05)


def get_values_and_probs_from_enum(enum):
    return list(zip(*map(lambda entry: (entry[0], entry[1]), map(lambda x: x.value, enum._member_map_.values()))))


def draw(enum):
    """
    Draws a random value from the given enum.
    """
    values, probs = get_values_and_probs_from_enum(enum)
    return int(np.random.choice(values, p=probs))


def generate_starting_lane(row):
    """Generates a dict with all starting lane information"""
    lane_dict = {'row': row,
                 'type': StartLane.__name__
                 }
    return lane_dict


def generate_finish_lane(row):
    """Generates a dict with all finish lane information"""
    lane_dict = {'row': row,
                 'type': FinishLane.__name__
                 }
    return lane_dict


def generate_normal_lane(row):
    """
    Generates a normal lane and returns it.
    """
    lane_dict = {'row': row,
                 'type': Lane.__name__, }
    return lane_dict


def generate_street_lane(row):
    """
    Generates a street lane and returns it.
    """
    lane_dict = {'row': row,
                 'type': StreetLane.__name__,
                 'direction': np.random.choice([LaneDirection.LEFT, LaneDirection.RIGHT]).value,
                 'lane_velocity': draw(LaneVelocity),
                 'obstacle_size': draw(VehicleWidth),
                 'distance_between_obstacles': draw(DistanceBetweenObstacles),
                 'obstacles_without_gap': draw(ObstaclesWithoutGap)}

    return lane_dict


def generate_water_lane(row):
    """
    Generates a water lane and returns it.
    """
    lane_dict = {'row': row,
                 'type': WaterLane.__name__,
                 'direction': np.random.choice([LaneDirection.LEFT, LaneDirection.RIGHT]).value,
                 'lane_velocity': draw(LaneVelocity),
                 'obstacle_size': draw(LilyPadWidth),
                 'distance_between_obstacles': draw(DistanceBetweenObstacles),
                 'obstacles_without_gap': draw(ObstaclesWithoutGap)}

    return lane_dict


def generate_world():
    """
    Generates a world and returns it.
    """
    world_dict = {}
    height = config.N_LANES
    width = config.N_FIELDS_PER_LANE

    # TODO also draw probabilistically?
    n_street_lanes = config.N_STREET_LANES
    n_middle_lanes = 1
    n_water_lanes = config.N_WATER_LANES

    world_dict['width'] = width
    world_dict['height'] = height
    world_dict['street_lanes'] = n_street_lanes
    world_dict['middle_lanes'] = n_middle_lanes
    world_dict['water_lanes'] = n_water_lanes

    # starting position
    if config.N_FIELDS_PER_LANE % 2 == 0:
        starting_position_x = width // 2 - 1
    else:
        starting_position_x = width // 2
    starting_position_y = height - 1
    world_dict['starting_position'] = (starting_position_x, starting_position_y)

    # target position
    target_position = TARGET_POSITIONS[draw(TargetPositionX)]
    world_dict['target_position'] = target_position
    lanes = []

    row = 0
    # create finish lane
    finish_lane = generate_finish_lane(row)
    lanes.append(finish_lane)
    row += 1

    # create water lanes
    for i in range(n_water_lanes):
        water_lane = generate_water_lane(row)
        lanes.append(water_lane)
        row += 1

    # create interim lanes
    interim_lane = generate_normal_lane(row)
    lanes.append(interim_lane)
    row += 1

    # create street lanes
    for i in range(n_street_lanes):
        street_lane = generate_street_lane(row)
        lanes.append(street_lane)
        row += 1

    # create start lane
    start_lane = generate_starting_lane(row)
    lanes.append(start_lane)

    assert row == height - 1, "number of rows is {}, but expected {}".format(row, height - 1)

    world_dict['lanes'] = lanes
    return world_dict


world_generation_utils.save_world_dict(generate_world(), "random_world")
