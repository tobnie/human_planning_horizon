import random
from enum import Enum

import config
from world.field import Field


class LaneType(Enum):
    START = 0
    STREET = 1
    INTERMEDIATE = 2
    WATER = 3
    END = 4


class LaneDirection(Enum):
    LEFT = 0
    RIGHT = 1


class Lane:

    def __init__(self, row: int):
        self.row = row  # row number in the game (counting from the top)
        self.fields = [Field(i, self.row) for i in range(config.N_FIELDS_PER_LANE)]


class StartLane(Lane):

    def __init__(self, row: int, starting_position: (float, float)):
        super().__init__(row)

        # TODO how to represent. Just an integer for predefined start_positions or a certain Point?
        self.starting_position = starting_position


class FinishLane(Lane):

    def __init__(self, row: int, end_position: (float, float)):
        super().__init__(row)

        # TODO how to represent. Just an integer for predefined start_positions or a certain Point? (same as with StartLane)
        self.end_position = end_position


class DirectedLane(Lane):

    def __init__(self, row: int, lane_direction: LaneDirection = None):
        super().__init__(row)

        # take given direction, otherwise choose randomly if left or right
        self.direction = lane_direction if lane_direction is not None else random.choice([LaneDirection.LEFT, LaneDirection.RIGHT])


class StreetLane(DirectedLane):

    def __init__(self, row: int, lane_direction: LaneDirection = None):
        super().__init__(row, lane_direction)


class WaterLane(DirectedLane):

    def __init__(self, row: int, lane_direction: LaneDirection = None):
        super().__init__(row, lane_direction)
