from collections import namedtuple
from enum import Enum

DiscreteDistribution = namedtuple('DiscreteDistribution', ['values', 'probabilities'])

# Target Positions
TARGET_POSITIONS = [2, 10, 18]


# Game Difficulty
class GameDifficulty(Enum):
    EASY = 'easy'
    NORMAL = 'normal'
    HARD = 'hard'


class GameParameter(Enum):
    LaneVelocity = 0
    VehicleWidth = 1
    LilyPadWidth = 2
    DistanceBetweenObstacles = 3    # TODO also split this for LilyPads and Vehicles?
    VehicleSpawnGap = 4
    LilyPadSpawnGap = 5
    TargetPositionX = 6


# Lane Velocities
LaneVelocityEasy = DiscreteDistribution(values=[1, 2], probabilities=[0.8, 0.2])
LaneVelocityNormal = DiscreteDistribution(values=[1, 2, 3], probabilities=[0.3, 0.5, 0.2])
LaneVelocityHard = DiscreteDistribution(values=[1, 2, 3, 4], probabilities=[0.1, 0.5, 0.35, 0.05])

LaneVelocities = {
    GameDifficulty.EASY: LaneVelocityEasy,
    GameDifficulty.NORMAL: LaneVelocityNormal,
    GameDifficulty.HARD: LaneVelocityHard,
}

# Vehicle Widths
VehicleWidthEasy = DiscreteDistribution(values=[1, 2], probabilities=[0.8, 0.2])
VehicleWidthNormal = DiscreteDistribution(values=[1, 2, 3], probabilities=[0.3, 0.5, 0.2])
VehicleWidthHard = DiscreteDistribution(values=[1, 2, 3], probabilities=[0.1, 0.55, 0.35])

VehicleWidths = {
    GameDifficulty.EASY: VehicleWidthEasy,
    GameDifficulty.NORMAL: VehicleWidthNormal,
    GameDifficulty.HARD: VehicleWidthHard
}

# Lily Pad Widths
LilyPadWidthEasy = DiscreteDistribution(values=[2, 3, 4], probabilities=[.1, .45, .45])
LilyPadWidthNormal = DiscreteDistribution(values=[1, 2, 3, 4], probabilities=[.1, .45, .4, .05])
LilyPadWidthHard = DiscreteDistribution(values=[1, 2], probabilities=[.4, .6])
LilyPadWidths = {
    GameDifficulty.EASY: LilyPadWidthEasy,
    GameDifficulty.NORMAL: LilyPadWidthNormal,
    GameDifficulty.HARD: LilyPadWidthHard
}

# Target Positions
TargetPositions = DiscreteDistribution(values=TARGET_POSITIONS, probabilities=[1 / len(TARGET_POSITIONS)] * len(TARGET_POSITIONS))

# DistanceBetweenObstacles
DistanceBetweenObstaclesEasy = DiscreteDistribution(values=[2, 3, 4], probabilities=[0.2, 0.5, 0.3])
DistanceBetweenObstaclesNormal = DiscreteDistribution(values=[1, 2, 3], probabilities=[0.2, 0.5, 0.3])
DistanceBetweenObstaclesHard = DiscreteDistribution(values=[1, 2], probabilities=[0.4, 0.6])
DistanceBetweenObstacles = {
    GameDifficulty.EASY: DistanceBetweenObstaclesEasy,
    GameDifficulty.NORMAL: DistanceBetweenObstaclesNormal,
    GameDifficulty.HARD: DistanceBetweenObstaclesHard
}

# Spawn Gaps (after how many obstacles should there be a gap)
LilyPadSpawnGapsEasy = DiscreteDistribution(values=[3, 4], probabilities=[.2, .8])
LilyPadSpawnGapsNormal = DiscreteDistribution(values=[2, 3, 4], probabilities=[.1, .5, .4])
LilyPadSpawnGapsHard = DiscreteDistribution(values=[1, 2, 3], probabilities=[.1, .5, .4])
LilyPadSpawnGaps = {
    GameDifficulty.EASY: LilyPadSpawnGapsEasy,
    GameDifficulty.NORMAL: LilyPadSpawnGapsNormal,
    GameDifficulty.HARD: LilyPadSpawnGapsHard
}

# Spawn Gaps Vehicles
VehicleSpawnGapsEasy = DiscreteDistribution(values=[1, 2], probabilities=[.2, .8])
VehicleSpawnGapsNormal = DiscreteDistribution(values=[2, 3, 4], probabilities=[.1, .5, .4])
VehicleSpawnGapsHard = DiscreteDistribution(values=[3, 4], probabilities=[.2, .8])
VehicleSpawnGaps = {
    GameDifficulty.EASY: VehicleSpawnGapsEasy,
    GameDifficulty.NORMAL: VehicleSpawnGapsNormal,
    GameDifficulty.HARD: VehicleSpawnGapsHard
}

ParameterDistributions = {
    GameParameter.LaneVelocity: LaneVelocities,
    GameParameter.VehicleWidth: VehicleWidths,
    GameParameter.LilyPadWidth: LilyPadWidths,
    GameParameter.DistanceBetweenObstacles: DistanceBetweenObstacles,
    GameParameter.TargetPositionX: TargetPositions,
    GameParameter.LilyPadSpawnGaps: LilyPadSpawnGaps,
    GameParameter.VehicleSpawnGaps: VehicleSpawnGaps
}
