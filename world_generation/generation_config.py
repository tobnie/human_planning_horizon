from collections import namedtuple
from enum import Enum

DiscreteDistribution = namedtuple('DiscreteDistribution', ['values', 'probabilities'])

# Target Positions
TARGET_POSITIONS = [2, 9, 17]


# Game Difficulty
class GameDifficulty(Enum):
    EASY = 'easy'
    NORMAL = 'normal'
    HARD = 'hard'


class GameParameter(Enum):
    LaneVelocity = 0
    VehicleWidth = 1
    LilyPadWidth = 2
    DistanceBetweenObstaclesLilyPad = 3
    DistanceBetweenObstaclesVehicle = 4
    VehicleSpawnProbability = 5
    LilyPadSpawnProbability = 6
    TargetPosition = 7


# Lane Velocities
LaneVelocityEasy = DiscreteDistribution(values=[1, 2], probabilities=[0.8, 0.2])
LaneVelocityNormal = DiscreteDistribution(values=[1, 2, 4], probabilities=[0.3, 0.5, 0.2])
LaneVelocityHard = DiscreteDistribution(values=[1, 2, 4, 8], probabilities=[0.1, 0.5, 0.35, 0.05])

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
LilyPadWidthHard = DiscreteDistribution(values=[1, 2, 3], probabilities=[.35, .55, .1])
LilyPadWidths = {
    GameDifficulty.EASY: LilyPadWidthEasy,
    GameDifficulty.NORMAL: LilyPadWidthNormal,
    GameDifficulty.HARD: LilyPadWidthHard
}

# Target Positions
TargetPositionsEasy = TargetPositionsMedium = TargetPositionsHard = DiscreteDistribution(values=TARGET_POSITIONS, probabilities=[1 / len(TARGET_POSITIONS)] * len(TARGET_POSITIONS))
TargetPositions = {
    GameDifficulty.EASY: TargetPositionsEasy,
    GameDifficulty.NORMAL: TargetPositionsMedium,
    GameDifficulty.HARD: TargetPositionsHard
}

# DistanceBetweenObstacles LilyPad
DistanceBetweenObstaclesLilyPadEasy = DiscreteDistribution(values=[1, 2], probabilities=[0.2, 0.8])
DistanceBetweenObstaclesLilyPadNormal = DiscreteDistribution(values=[1, 2, 3], probabilities=[0.05, 0.35, 0.6])
DistanceBetweenObstaclesLilyPadHard = DiscreteDistribution(values=[3, 4], probabilities=[0.55, 0.45])
DistanceBetweenObstaclesLilyPad = {
    GameDifficulty.EASY: DistanceBetweenObstaclesLilyPadEasy,
    GameDifficulty.NORMAL: DistanceBetweenObstaclesLilyPadNormal,
    GameDifficulty.HARD: DistanceBetweenObstaclesLilyPadHard
}

# Distance Between Obstacles Vehicle
DistanceBetweenObstaclesVehicleEasy = DiscreteDistribution(values=[-1, 3, 4], probabilities=[0.3, 0.4, 0.3])
DistanceBetweenObstaclesVehicleNormal = DiscreteDistribution(values=[1, 2, 3], probabilities=[0.2, 0.5, 0.3])
DistanceBetweenObstaclesVehicleHard = DiscreteDistribution(values=[1, 2], probabilities=[0.5, 0.5])
DistanceBetweenObstaclesVehicle = {
    GameDifficulty.EASY: DistanceBetweenObstaclesVehicleEasy,
    GameDifficulty.NORMAL: DistanceBetweenObstaclesVehicleNormal,
    GameDifficulty.HARD: DistanceBetweenObstaclesVehicleHard
}

# # Spawn Gaps (after how many obstacles should there be a gap)
# LilyPadSpawnGapsEasy = DiscreteDistribution(values=[3, 4], probabilities=[.2, .8])
# LilyPadSpawnGapsNormal = DiscreteDistribution(values=[2, 3, 4], probabilities=[.1, .5, .4])
# LilyPadSpawnGapsHard = DiscreteDistribution(values=[1, 2, 3], probabilities=[.1, .5, .4])
# LilyPadSpawnGaps = {
#     GameDifficulty.EASY: LilyPadSpawnGapsEasy,
#     GameDifficulty.NORMAL: LilyPadSpawnGapsNormal,
#     GameDifficulty.HARD: LilyPadSpawnGapsHard
# }
#
# # Spawn Gaps Vehicles
# VehicleSpawnGapsEasy = DiscreteDistribution(values=[1, 2], probabilities=[.2, .8])
# VehicleSpawnGapsNormal = DiscreteDistribution(values=[2, 3, 4], probabilities=[.1, .5, .4])
# VehicleSpawnGapsHard = DiscreteDistribution(values=[3, 4], probabilities=[.2, .8])
# VehicleSpawnGaps = {
#     GameDifficulty.EASY: VehicleSpawnGapsEasy,
#     GameDifficulty.NORMAL: VehicleSpawnGapsNormal,
#     GameDifficulty.HARD: VehicleSpawnGapsHard
# }

LilyPadSpawnProbabilityEasy = 0.9
LilyPadSpawnProbabilityNormal = 0.7
LilyPadSpawnProbabilityHard = 0.5
LilyPadSpawnProbability = {
    GameDifficulty.EASY: LilyPadSpawnProbabilityEasy,
    GameDifficulty.NORMAL: LilyPadSpawnProbabilityNormal,
    GameDifficulty.HARD: LilyPadSpawnProbabilityHard
}

VehicleSpawnProbabilityEasy = 0.5
VehicleSpawnProbabilityNormal = 0.7
VehicleSpawnProbabilityHard = 0.9
VehicleSpawnProbability = {
    GameDifficulty.EASY: VehicleSpawnProbabilityEasy,
    GameDifficulty.NORMAL: VehicleSpawnProbabilityNormal,
    GameDifficulty.HARD: VehicleSpawnProbabilityHard
}

ParameterDistributions = {
    GameParameter.LaneVelocity: LaneVelocities,
    GameParameter.VehicleWidth: VehicleWidths,
    GameParameter.LilyPadWidth: LilyPadWidths,
    GameParameter.DistanceBetweenObstaclesLilyPad: DistanceBetweenObstaclesLilyPad,
    GameParameter.DistanceBetweenObstaclesVehicle: DistanceBetweenObstaclesVehicle,
    GameParameter.TargetPosition: TargetPositions,
    GameParameter.LilyPadSpawnProbability: LilyPadSpawnProbability,
    GameParameter.VehicleSpawnProbability: VehicleSpawnProbability
}
