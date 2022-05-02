import random
from abc import ABC
from enum import Enum

import pygame.sprite

import colors
import config
from world.dynamic_object import Vehicle, LilyPad, DynamicObject
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
        rect_height = config.MONITOR_HEIGHT_PX / config.N_LANES  # TODO ADJUST HEIGHT ?
        self.rect = pygame.Rect(0, self.row * rect_height, config.MONITOR_WIDTH_PX, rect_height)
        self.fields = [Field(i, self.row) for i in range(config.N_FIELDS_PER_LANE)]
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

    def spawn_entity(self) -> None:
        new_entity: DynamicObject = self.object_type()

        entity_rect = new_entity.image.get_rect()

        # spawn position (just not visible out of screen)
        if self.direction == LaneDirection.RIGHT:
            new_entity.rect.x = - entity_rect.width
            new_entity.delta_x = 1
        else:
            new_entity.rect.x = self.rect.width + entity_rect.width
            new_entity.delta_x = -1
        new_entity.rect.y = self.row * self.rect.height

        self.sprites.add(new_entity)

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
