from abc import ABC
from enum import Enum

import numpy as np
import pygame.sprite

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
        self.fields = [Field(i, self.row, color, config.SPRITES_DIR + 'grass.png') for i in range(config.N_FIELDS_PER_LANE)]

    def draw_lane(self, screen):
        for field in self.fields:
            field.draw(screen)

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

        # draw star on target field
        img_path = config.SPRITES_DIR + 'star.jpg'

        img = pygame.image.load(img_path)
        img.set_colorkey(colors.WHITE)
        img.convert_alpha()
        img = pygame.transform.scale(img, [config.FIELD_WIDTH, config.FIELD_HEIGHT])
        screen.blit(img, (target_field.rect.x, target_field.rect.y))


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
        self.currently_in_spawn_gap = False
        self.obstacle_counter = 0

        # counts the calls of the update method and only updates the sprites if the update count is a multiple of the velocity
        self.update_cnt = 0

    def calc_distance_of_new_to_last_sprite(self) -> int:
        """
        Calculates distance to the last sprite in the lane if a new sprite was spawned now.
        :return: distance in fields
        """
        if len(self.non_player_sprites) == 0:
            return np.inf
        else:
            if self.direction == LaneDirection.LEFT:
                obstacle = self.non_player_sprites.sprites()[-1]
                return len(self) - (obstacle.x + obstacle.width)
            else:
                return self.non_player_sprites.sprites()[-1].x

    def spawn_entity(self) -> None:
        """ Spawns a new entity in the lane conforming to the currently active entities and the specified lane parameters. """
        if self.obstacle_counter == self.obstacles_without_gap:
            self.currently_in_spawn_gap = True

        # if currently in spawn gap, do not spawn new sprite and increase threshold for next spawn
        threshold = self.distance_between_obstacles if not self.currently_in_spawn_gap else 2 * self.distance_between_obstacles + self.obstacle_size

        # if distance to last sprite is smaller than threshold, do not spawn new sprite
        if self.calc_distance_of_new_to_last_sprite() >= threshold:
            if self.currently_in_spawn_gap:
                self.currently_in_spawn_gap = False
                self.obstacle_counter = 0
            # skip obstacle if spawn gap is reached
            if not self.currently_in_spawn_gap:
                # get obstacle parameters
                if self.direction == LaneDirection.RIGHT:
                    obstacle_velocity = self.velocity
                    obstacle_x = -self.obstacle_size
                else:
                    obstacle_velocity = -self.velocity
                    obstacle_x = len(self)

                # create new obstacles
                new_entity: DynamicObject = self.dynamic_object_constructor(self.world, obstacle_x, self.row, obstacle_velocity,
                                                                            self.obstacle_size, 1)
                self.non_player_sprites.add(new_entity)
                self.obstacle_counter += 1

    def update(self) -> None:

        if self.update_cnt != self.velocity:
            self.update_cnt += 1
            return

        self.non_player_sprites.update()

        self.update_cnt = 0

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
        self.fields = [Field(i, self.row, self.color, None) for i in range(config.N_FIELDS_PER_LANE)]


class WaterLane(DirectedLane):

    def __init__(self, world, row: int, lane_direction: LaneDirection = LaneDirection.LEFT, velocity: int = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1, obstacles_without_gap: int = 3):
        super().__init__(world, row, lane_direction, colors.BLUE, velocity,
                         distance_between_obstacles, obstacle_size, obstacles_without_gap)
        self.dynamic_object_constructor = LilyPad
        self.fields = [Field(i, self.row, self.color, config.SPRITES_DIR + 'water.png') for i in range(config.N_FIELDS_PER_LANE)]
