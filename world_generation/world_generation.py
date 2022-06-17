from enum import Enum


class ProbabilityEnum(Enum):

    def __init__(self, value, probability):
        self.value = value
        self.probability = probability


class LaneVelocity(ProbabilityEnum):
    """
    Enum for lane velocities.
    """
    SLOW = 1, 0.6
    MEDIUM = 2, 0.3
    FAST = 3, 0.1


class VehicleWidth(ProbabilityEnum):
    """
    Enum for vehicle widths.
    """
    SMALL = 1, 0.7
    MEDIUM = 2, 0.2
    LARGE = 3, 0.1


class LilyPadWidth(ProbabilityEnum):
    """
    Enum for lilypad widths.
    """
    SMALL = 1, 0.05
    MEDIUM = 2, 0.35
    LARGE = 3, 0.4
    XLarge = 4, 0.2


class TargetPositionX(ProbabilityEnum):
    """
    Enum for target position X.
    """
    LEFT = 1, 1 / 3
    MIDDLE = 2, 1 / 3
    RIGHT = 3, 1 / 3


class DistanceBetweenObstacles(ProbabilityEnum):
    """
    Enum for distance between obstacles.
    """
    # TODO even allow SMALL = 1 Field distances?
    SMALL = 1, 0.1
    MEDIUM = 2, 0.4
    LARGE = 3, 0.4
    XLarge = 4, 0.1


def generate_world():
    """
    Generates a world and returns it.
    """
    # TODO
    pass
