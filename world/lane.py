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

    def __init__(self, row: int, color: (int, int, int) = colors.GREEN):
        super().__init__()
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

    def __init__(self, row: int, starting_position: (int, int)):
        super().__init__(row, colors.RED)
        self.starting_position = starting_position


class FinishLane(Lane):

    def __init__(self, row: int, end_position: (int, int)):
        super().__init__(row, colors.RED)
        self.end_position = end_position


class DirectedLane(Lane, ABC):

    def __init__(self, row: int, lane_direction: LaneDirection = None, color: (int, int, int) = colors.GREEN):
        super().__init__(row, color)

        # take given direction, otherwise choose randomly if left or right
        if not lane_direction:
            self.direction = LaneDirection.LEFT if row % 2 == 0 else LaneDirection.RIGHT
        else:
            self.direction = lane_direction
        self.non_player_sprites = pygame.sprite.Group()
        self.dynamic_object_constructor = None

        # average spawn rate should be time for a movement covering the whole width of the current object times two
        # time = width / velocity
        self.spawn_counter = 0
        self.next_spawn_event = 0
        self.next_obstacle_size = 0
        self.time_between_spawns = 0
        self._reset_spawn_counter()

    def _reset_spawn_counter(self):
        # TODO for generated worlds
        self.spawn_counter = 0
        self.next_obstacle_size = random.choice(config.OBSTACLE_WIDTH)
        self.time_between_spawns = random.choice(
            self.next_obstacle_size / config.OBSTACLE_VELOCITY * np.array(config.OBSTACLE_SPAWN_TIME_MULTIPLIERS))
        self.next_spawn_event = 5
        # print(f"self.next_spawn_event={self.next_spawn_event}")
        # print(f"self.time_between_spawns={self.time_between_spawns}")
        # print(f"self.next_obstacle_size={self.next_obstacle_size}")
        # print(f"self.next_spawn_event={self.next_spawn_event}")

    def spawn_entity(self) -> None:
        # if counter is over time for next event, spawn entity
        if self.spawn_counter > self.next_spawn_event:
            if self.direction == LaneDirection.RIGHT:
                obst_delta_x = 1
                spawn_x = -1
            else:
                obst_delta_x = -1
                spawn_x = len(self)

            new_entity: DynamicObject = self.dynamic_object_constructor(spawn_x, self.row,
                                                                        self.next_obstacle_size, 1,
                                                                        delta_x=obst_delta_x)

            # spawn entity behind if it overlaps with current last entity
            if self.non_player_sprites.sprites():
                last_sprite = self.non_player_sprites.sprites()[0]
                if collision_handler.check_collision(new_entity, last_sprite):
                    if self.direction == LaneDirection.RIGHT:
                        new_entity.x = last_sprite.rect.x - new_entity.rect.width
                    else:
                        new_entity.x = last_sprite.rect.x + last_sprite.rect.width
                    new_entity.rect.x = new_entity.x

            self.non_player_sprites.add(new_entity)
            self._reset_spawn_counter()
        else:
            self.spawn_counter += 1

    def update(self) -> None:
        self.non_player_sprites.update()

    def draw_lane(self, screen) -> None:
        super().draw_lane(screen)
        self.non_player_sprites.draw(screen)


class StreetLane(DirectedLane):

    def __init__(self, row: int, lane_direction: LaneDirection = LaneDirection.LEFT):
        super().__init__(row, lane_direction, colors.BLACK)
        self.dynamic_object_constructor = Vehicle


class WaterLane(DirectedLane):

    def __init__(self, row: int, lane_direction: LaneDirection = LaneDirection.LEFT):
        super().__init__(row, lane_direction, colors.BLUE)
        self.dynamic_object_constructor = LilyPad
