import random

import numpy as np
import pygame

from generation_config import GameDifficulty, ParameterDistributions, GameParameter

pygame.init()

from world.lane import LaneDirection, StartLane, FinishLane, Lane, StreetLane, WaterLane
import world_generation_utils
import config


class WorldGenerator:

    def __init__(self, path=config.LEVELS_DIR):
        self.difficulty = None
        self.base_directory_path = path
        self.save_directory = self.base_directory_path + "undefined" + '/'

    def _set_difficulty(self, difficulty: GameDifficulty):
        self.difficulty = difficulty
        self.save_directory = self.base_directory_path + difficulty.value + '/'

    def _draw(self, parameter: GameParameter):
        """
        Draws a random value for the given game parameter
        """
        parameter_distribution = ParameterDistributions[parameter][self.difficulty]
        values, probs = parameter_distribution.values, parameter_distribution.probabilities
        return int(np.random.choice(values, p=probs))

    def _generate_starting_lane(self, row):
        """Generates a dict with all starting lane information"""
        lane_dict = {'row': row,
                     'type': StartLane.__name__
                     }
        return lane_dict

    def _generate_finish_lane(self, row):
        """Generates a dict with all finish lane information"""
        lane_dict = {'row': row,
                     'type': FinishLane.__name__
                     }
        return lane_dict

    def _generate_normal_lane(self, row):
        """
        Generates a normal lane and returns it.
        """
        lane_dict = {'row': row,
                     'type': Lane.__name__, }
        return lane_dict

    def _generate_street_lane(self, row, direction):
        """
        Generates a street lane and returns it.
        """
        lane_dict = {'row': row,
                     'type': StreetLane.__name__,
                     'direction': LaneDirection.LEFT.value if direction == -1 else LaneDirection.RIGHT.value,
                     'lane_velocity': self._draw(GameParameter.LaneVelocity),
                     'obstacle_size': self._draw(GameParameter.VehicleWidth),
                     'distance_between_obstacles': self._draw(GameParameter.DistanceBetweenObstaclesVehicle),
                     'spawn_probability': ParameterDistributions[GameParameter.VehicleSpawnProbability][self.difficulty]}

        return lane_dict

    def _generate_water_lane(self, row, direction):
        """
        Generates a water lane and returns it.
        """
        lane_dict = {'row': row,
                     'type': WaterLane.__name__,
                     'direction': LaneDirection.LEFT.value if direction == -1 else LaneDirection.RIGHT.value,
                     'lane_velocity': self._draw(GameParameter.LaneVelocity),
                     'obstacle_size': self._draw(GameParameter.LilyPadWidth),
                     'distance_between_obstacles': self._draw(GameParameter.DistanceBetweenObstaclesLilyPad),
                     'spawn_probability': ParameterDistributions[GameParameter.LilyPadSpawnProbability][self.difficulty]}

        return lane_dict

    def generate_and_save_world(self, difficulty: GameDifficulty, world_name: str):
        """
        Generates a world and saves it to a file.
        """
        world_dict = self.generate_world(difficulty)
        world_generation_utils.save_world_dict(world_dict, world_name, path=self.save_directory)

    def generate_world(self, difficulty: GameDifficulty):
        """
        Generates a world and returns it.
        """
        self._set_difficulty(difficulty)

        world_dict = {}
        height = config.N_LANES
        width = config.N_FIELDS_PER_LANE

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
        target_position = self._draw(GameParameter.TargetPosition)
        world_dict['target_position'] = target_position
        lanes = []

        row = 0
        # create finish lane
        finish_lane = self._generate_finish_lane(row)
        lanes.append(finish_lane)
        row += 1

        # determine movement direction of first lane
        direction = random.choice([-1, 1])

        # create water lanes
        for i in range(n_water_lanes):
            water_lane = self._generate_water_lane(row, direction)
            lanes.append(water_lane)
            direction *= -1
            row += 1

        # create interim lanes
        interim_lane = self._generate_normal_lane(row)
        lanes.append(interim_lane)
        row += 1

        # create street lanes
        for i in range(n_street_lanes):
            street_lane = self._generate_street_lane(row, direction)
            lanes.append(street_lane)
            direction *= -1
            row += 1

        # create start lane
        start_lane = self._generate_starting_lane(row)
        lanes.append(start_lane)

        assert row == height - 1, "number of rows is {}, but expected {}".format(row, height - 1)

        world_dict['lanes'] = lanes
        return world_dict
