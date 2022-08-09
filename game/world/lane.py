from abc import ABC
from enum import Enum

import numpy as np
import pygame.sprite

import colors
import config
from game.world.game_object import Vehicle, LilyPad, Obstacle
from game.world.field import Field


class LaneType(Enum):
    START = 0
    STREET = 1
    INTERMEDIATE = 2
    WATER = 3
    END = 4


class LaneDirection(Enum):
    LEFT = -1
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
        img_path = config.SPRITES_DIR + config.STAR_FILE

        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, [config.FIELD_WIDTH, config.FIELD_HEIGHT])
        img.set_colorkey(colors.WHITE)
        img.convert_alpha()
        screen.blit(img, (target_field.rect.x, target_field.rect.y))


class DirectedLane(Lane, ABC):

    def __init__(self, world, row: int, lane_direction: LaneDirection = None, color: (int, int, int) = colors.GREEN, velocity: float = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1,
                 spawn_probability: float = 0.5):
        super().__init__(world, row, color)

        # take given direction, otherwise choose randomly if left or right
        if not lane_direction:
            self.direction = LaneDirection.LEFT if row % 2 == 0 else LaneDirection.RIGHT
        else:
            self.direction = lane_direction
        self.non_player_sprites = pygame.sprite.Group()
        self.obstacle_constructor = None

        # lane dynamics
        self.velocity = velocity
        self.direction = lane_direction
        self.base_distance_between_obstacles = distance_between_obstacles
        self.obstacle_size = obstacle_size
        self.spawn_probability = spawn_probability  # probability of spawning an obstacle in a lane
        self.missed_spawns = 0  # how many times no obstacle was spawned even though the distance threshold was reached

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
                dist_px = len(self) * config.FIELD_WIDTH - (obstacle.rect.x + obstacle.rect.width)
                return dist_px
            else:
                return self.non_player_sprites.sprites()[-1].rect.x

    def spawn_entity(self) -> None:
        """ Spawns a new entity in the lane conforming to the currently active entities and the specified lane parameters. """

        # if the distance is set to be infinity, never spawn anything
        if self.base_distance_between_obstacles == -1:
            return

        threshold = self.base_distance_between_obstacles + self.missed_spawns * (self.base_distance_between_obstacles + self.obstacle_size)

        # if distance to last sprite is smaller than threshold, do not spawn new sprite
        if self.calc_distance_of_new_to_last_sprite() >= threshold * config.FIELD_WIDTH:

            if np.random.random() < self.spawn_probability:
                # get obstacle parameters
                if self.direction == LaneDirection.RIGHT:
                    obstacle_velocity = self.velocity
                    obstacle_x = -self.obstacle_size * config.FIELD_WIDTH
                else:
                    obstacle_velocity = -self.velocity
                    obstacle_x = len(self) * config.FIELD_WIDTH

                # create new obstacle
                obstacle_y = self.row * config.FIELD_HEIGHT
                new_entity: Obstacle = self.obstacle_constructor(self.world, obstacle_x, obstacle_y, obstacle_velocity,
                                                                 self.obstacle_size, 1)
                self.non_player_sprites.add(new_entity)
                self.missed_spawns = 0
            else:
                self.missed_spawns += 1

    def draw_lane(self, screen) -> None:
        """ Draws the lane and all its entities on the given screen object.  """
        super().draw_lane(screen)
        self.non_player_sprites.draw(screen)

    def update(self) -> None:
        """ Updates the position of all obstacles in the lane. """
        for obstacle in self.non_player_sprites:
            obstacle.update()


class StreetLane(DirectedLane):

    def __init__(self, world, row: int, lane_direction: LaneDirection = LaneDirection.LEFT, velocity: float = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1, spawn_probability: int = 3
                 ):
        super().__init__(world, row, lane_direction, colors.BLACK, velocity,
                         distance_between_obstacles, obstacle_size, spawn_probability)

        # set distance between vehicles to infinity if distance is coded as -1 in json (i.e. no obstacles)
        if distance_between_obstacles == -1:
            self.distance_between_obstacles = np.inf

        self.obstacle_constructor = Vehicle
        self.fields = [Field(i, self.row, self.color, None) for i in range(config.N_FIELDS_PER_LANE)]


class WaterLane(DirectedLane):

    def __init__(self, world, row: int, lane_direction: LaneDirection = LaneDirection.LEFT, velocity: float = 1,
                 distance_between_obstacles: int = 4, obstacle_size: int = 1, spawn_probability: int = 3):
        super().__init__(world, row, lane_direction, colors.BLUE, velocity,
                         distance_between_obstacles, obstacle_size, spawn_probability)
        self.obstacle_constructor = LilyPad
        self.fields = [Field(i, self.row, self.color, config.SPRITES_DIR + 'water.png') for i in range(config.N_FIELDS_PER_LANE)]
