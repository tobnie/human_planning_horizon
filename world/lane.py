import random
from abc import ABC
from enum import Enum

import numpy as np

import pygame.sprite

import collision_handler
import colors
import config
from world.dynamic_object import Vehicle, LilyPad, DynamicObject


class LaneType(Enum):
    START = 0
    STREET = 1
    INTERMEDIATE = 2
    WATER = 3
    END = 4


class LaneDirection(Enum):
    LEFT = 0
    RIGHT = 1


class Lane(pygame.sprite.Sprite):

    def __init__(self, row: int, color: (int, int, int) = colors.GREEN):
        super().__init__()
        self.row = row  # row number in the game (counting from the top)
        rect_height = config.MONITOR_HEIGHT_PX / config.N_LANES  # TODO ADJUST HEIGHT ?
        self.rect = pygame.Rect(0, self.row * rect_height, config.MONITOR_WIDTH_PX, rect_height)
        self.color = color

    def draw_lane(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)


class StartLane(Lane):

    def __init__(self, row: int, starting_position: int):
        super().__init__(row, colors.RED)

        # TODO how to represent. Just an integer for predefined start_positions or a certain Point?
        self.starting_position = starting_position


class FinishLane(Lane):

    def __init__(self, row: int, end_position: (float, float)):
        super().__init__(row, colors.RED)

        # TODO how to represent. Just an integer for predefined start_positions or a certain Point? (same as with StartLane)
        self.end_position = end_position


class DirectedLane(Lane, ABC):

    def __init__(self, row: int, lane_direction: LaneDirection = None, color: (int, int, int) = colors.GREEN):
        super().__init__(row, color)

        # take given direction, otherwise choose randomly if left or right
        self.direction = lane_direction if lane_direction is not None else random.choice([LaneDirection.LEFT, LaneDirection.RIGHT])
        self.sprites = pygame.sprite.Group()
        self.object_type = None

        # average spawn rate should be time for a movement covering the whole width of the current object times two
        # time = width / velocity
        self.spawn_counter = 0
        self.next_spawn_event = 0
        self.next_obstacle_size = 0
        self.time_between_spawns = 0
        self._reset_spawn_counter()

    def _reset_spawn_counter(self):
        self.spawn_counter = 0
        self.next_obstacle_size = config.OBSTACLE_WIDTH * random.choice(config.OBSTACLE_WIDTH_MULTIPLIERS)
        self.time_between_spawns = random.choice(
            self.next_obstacle_size / config.OBSTACLE_VELOCITY * np.array(config.OBSTACLE_SPAWN_TIME_MULTIPLIERS))
        self.next_spawn_event = np.random.poisson(self.time_between_spawns)
        print(f"self.next_spawn_event={self.next_spawn_event}")
        print(f"self.time_between_spawns={self.time_between_spawns}")
        print(f"self.next_obstacle_size={self.next_obstacle_size}")
        print(f"self.next_spawn_event={self.next_spawn_event}")

    def spawn_entity(self) -> None:
        # if counter is over time for next event, spawn entity
        if self.spawn_counter > self.next_spawn_event:
            if self.direction == LaneDirection.RIGHT:
                obst_delta_x = config.OBSTACLE_VELOCITY
                spawn_x = - self.next_obstacle_size
            else:
                obst_delta_x = -config.OBSTACLE_VELOCITY
                spawn_x = self.rect.width

            new_entity: DynamicObject = self.object_type(spawn_x, self.row * self.rect.height,
                                                         self.next_obstacle_size, self.rect.height,
                                                         delta_x=obst_delta_x)

            # spawn entity behind if it overlaps with current last entity
            if self.sprites.sprites():
                last_sprite = self.sprites.sprites()[0]
                if collision_handler.check_collision(new_entity, last_sprite):
                    if self.direction == LaneDirection.RIGHT:
                        new_entity.x = last_sprite.rect.x - new_entity.rect.width
                    else:
                        new_entity.x = last_sprite.rect.x + last_sprite.rect.width
                    new_entity.rect.x = new_entity.x

            self.sprites.add(new_entity)
            self._reset_spawn_counter()
        else:
            self.spawn_counter += 1

    def update(self) -> None:
        self.sprites.update()

    def draw_lane(self, screen) -> None:
        pygame.draw.rect(screen, self.color, pygame.Rect((0, self.row * self.rect.height), (self.rect.width, self.rect.height + 1)))
        self.sprites.draw(screen)


class StreetLane(DirectedLane):

    def __init__(self, row: int, lane_direction: LaneDirection = LaneDirection.LEFT):
        super().__init__(row, lane_direction, colors.BLACK)
        self.object_type = Vehicle


class WaterLane(DirectedLane):

    def __init__(self, row: int, lane_direction: LaneDirection = LaneDirection.LEFT):
        super().__init__(row, lane_direction, colors.BLUE)
        self.object_type = LilyPad
