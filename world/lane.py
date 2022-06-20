import random
from abc import ABC
from enum import Enum

import numpy as np

import pygame.sprite

import collision_handler
import colors
import config
from world.game_object import Vehicle, LilyPad, DynamicObject
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


class Lane(pygame.sprite.Sprite):

    def __init__(self, world, row: int, color: (int, int, int) = colors.GREEN):
        super().__init__()
        self.world = world
        self.row = row  # row number in the game (counting from the top)
        self.rect = pygame.Rect(0, self.row * config.FIELD_HEIGHT, config.DISPLAY_WIDTH_PX, config.FIELD_HEIGHT)
        self.color = color
        self.fields = [Field(i, self.row) for i in range(config.N_FIELDS_PER_LANE)]

    def draw_lane(self, screen):
        for field in self.fields:
            pygame.draw.rect(screen, self.color, field.rect)
            for i in range(4):
                pygame.draw.rect(screen, (0, 0, 0), (field.rect.x - i, field.rect.y - i, field.rect.width, field.rect.height), 1)

    def __len__(self):
        return len(self.fields)


class StartLane(Lane):

    def __init__(self, world, row: int, starting_position: (int, int)):
        super().__init__(world, row, colors.RED)
        self.starting_position = starting_position


class FinishLane(Lane):

    def __init__(self, world, row: int, target_position: (int, int)):
        super().__init__(world, row, colors.RED)
        self.target_position = target_position

    def draw_lane(self, screen):
        super().draw_lane(screen)
        # draw target position
        target_field = self.fields[self.target_position]
        for i in range(4):
            pygame.draw.rect(screen, colors.YELLOW,
                             (target_field.rect.x - i, target_field.rect.y - i, target_field.rect.width, target_field.rect.height), 1)


class DirectedLane(Lane, ABC):

    def __init__(self, world, row: int, lane_direction: LaneDirection = None, color: (int, int, int) = colors.GREEN, velocity: int = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1, obstacles_without_gap: int = 3):
        super().__init__(world, row, color)

        # take given direction, otherwise choose randomly if left or right
        if not lane_direction:
            self.direction = LaneDirection.LEFT if row % 2 == 0 else LaneDirection.RIGHT
        else:
            self.direction = lane_direction
        self.non_player_sprites = pygame.sprite.Group()
        self.dynamic_object_constructor = None

        # lane dynamics
        self.velocity = velocity
        self.direction = lane_direction
        self.distance_between_obstacles = distance_between_obstacles
        self.obstacle_size = obstacle_size
        self.obstacles_without_gap = obstacles_without_gap
        self.gap_counter = 0
        self.obstacle_counter = 0

    def spawn_entity(self) -> None:

        # if gap is complete, spawn next obstacle
        if self.gap_counter >= self.distance_between_obstacles:
            # skip obstacle if all obstacles with gap have been spawned
            if self.obstacle_counter == self.obstacles_without_gap:
                self.obstacle_counter = 0
                self.gap_counter = 0
            else:
                if self.direction == LaneDirection.RIGHT:
                    obst_delta_x = self.velocity
                    spawn_x = -1
                else:
                    obst_delta_x = -self.velocity
                    spawn_x = len(self)

                new_entity: DynamicObject = self.dynamic_object_constructor(self.world, spawn_x, self.row,
                                                                            self.obstacle_size, 1,
                                                                            delta_x=obst_delta_x)

                self.non_player_sprites.add(new_entity)
                self.gap_counter = 0
                self.obstacle_counter += 1
        else:
            self.gap_counter += 1

    def update(self) -> None:
        self.non_player_sprites.update()

    def draw_lane(self, screen) -> None:
        super().draw_lane(screen)
        self.non_player_sprites.draw(screen)


class StreetLane(DirectedLane):

    def __init__(self, world, row: int, lane_direction: LaneDirection = LaneDirection.LEFT, velocity: int = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1, obstacles_without_gap: int = 3
                 ):
        super().__init__(world, row, lane_direction, colors.BLACK, velocity,
                         distance_between_obstacles, obstacle_size, obstacles_without_gap)
        self.dynamic_object_constructor = Vehicle


class WaterLane(DirectedLane):

    def __init__(self, world, row: int, lane_direction: LaneDirection = LaneDirection.LEFT, velocity: int = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1, obstacles_without_gap: int = 3):
        super().__init__(world, row, lane_direction, colors.BLUE, velocity,
                         distance_between_obstacles, obstacle_size, obstacles_without_gap)
        self.dynamic_object_constructor = LilyPad
