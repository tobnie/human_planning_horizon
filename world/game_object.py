import os
import string
from abc import ABC

import pygame

import colors
import config


class GameObject(pygame.sprite.Sprite, ABC):
    """An abstract class for game objects"""

    def __init__(self, world, x: float = 0, y: int = 0, width: int = 1, height: int = 1, img_path: string = None):
        super().__init__()

        # assign object id
        self.id = id(self)
        self.world = world
        self.rotatable = True

        self.width = width
        self.height = height

        # sprite image
        if img_path:
            img = pygame.image.load(img_path).convert_alpha()
            img = pygame.transform.scale(img, [width * config.FIELD_WIDTH, height * config.FIELD_HEIGHT])
            self.base_img = img
            self.image = img
            self.image.set_colorkey(colors.WHITE)
            self.rect = self.image.get_rect()
        else:
            self.rect = pygame.Rect(x * config.FIELD_WIDTH, y * config.FIELD_HEIGHT, width * config.FIELD_WIDTH,
                                    height * config.FIELD_HEIGHT)

        # internal position (int)
        self.x: float = 0
        self.y: int = 0
        self.highest_visited_lane = 0
        self.set_position((x, y))

    def set_position(self, pos: (float, int)):
        """ Sets the current position of the player sprite rect given as (x, y)-tuple."""

        self.x = pos[0]
        self.y = pos[1]
        self.rect.x = int(round(pos[0]))
        self.rect.y = int(round(pos[1]))


class StaticObject(GameObject):
    """A class for static game objects, i.e. they have no ability to move."""

    def __init__(self, world, x: float = 0, y: int = 0, width: int = 1, height: int = 1, img_path: string = None):
        super().__init__(world, x, y, width, height, img_path=img_path)


class DynamicObject(GameObject, ABC):
    """A class for dynamic game objects, which are moving with a specified delta in either x- or y-direction."""

    def __init__(self, world, x: float = 0, y: int = 0, velocity: float = 0, width: int = 1, height: int = 1,
                 img_path: string = None,
                 movement_bounds_x: (int, int) = (0, config.DISPLAY_WIDTH_PX),
                 movement_bounds_y: (int, int) = (0, config.DISPLAY_HEIGHT_PX - config.FIELD_HEIGHT - 1)):
        super().__init__(world, x, y, width, height, img_path=img_path)

        self.velocity = velocity

        # dynamics
        self.movement_bounds_x = movement_bounds_x
        self.movement_bounds_y = movement_bounds_y


class Obstacle(DynamicObject):

    def __init__(self, world, x: float = 0, y: int = 0, velocity: float = 0, width: int = 1, height: int = 1,
                 img_path: string = None,
                 movement_bounds_x: (int, int) = (-1, config.N_FIELDS_PER_LANE),
                 movement_bounds_y: (int, int) = (-1, config.N_LANES), rotatable: bool = True):
        super().__init__(world, x, y, velocity, width, height, img_path=img_path, movement_bounds_x=movement_bounds_x,
                         movement_bounds_y=movement_bounds_y)

        if self.velocity == 0:
            raise ValueError("Velocity of an obstacle must not be 0.")

        self.delta_x = self.velocity // abs(self.velocity)
        self.rotatable = rotatable

        if self.rotatable:
            self.set_rotated_sprite_img()

    def set_rotated_sprite_img(self):
        """Sets the sprite image for the respective current direction of movement."""
        if self.velocity > 0:
            self.image = pygame.transform.rotate(self.base_img, 270)
        if self.velocity < 0:
            self.image = pygame.transform.rotate(self.base_img, 90)

        self.image = pygame.transform.scale(self.image, [self.rect.width, self.rect.height])
        self.image.set_colorkey(colors.WHITE)

    def update(self) -> None:
        """Updates the position of the obstacle."""
        new_x = self.x + self.delta_x
        self.set_position((new_x, self.y))


class Vehicle(Obstacle):
    """Vehicles are moving on the streets with specific properties given by the lane they are on (passed to the
    vehicle constructor upon spawning in the lane)"""

    def __init__(self, world, x: float = 0, y: int = 0, velocity: int = 0,
                 width: int = 1, height: int = 1):
        super().__init__(world, x, y, velocity, width, height, os.path.join(config.SPRITES_DIR, config.CAR_FILE))


class LilyPad(Obstacle):
    """Lilypads are moving on water with specific properties given by the lane they are on (passed to the
        lilypad constructor upon spawning in the lane)"""

    def __init__(self, world, x: float = 0, y: int = 0, velocity: int = 0,
                 width: int = 1, height: int = 1):
        if width == 1:
            img_file = config.LILYPAD_FILE
        else:
            img_file = config.LOG_FILE
        super().__init__(world, x, y, velocity, width, height, os.path.join(config.SPRITES_DIR, img_file), rotatable=False)

    def update(self) -> None:
        """Calls the super method as usual but also checks if the player is on the lilypad and if so,
        the player's position is also updated accordingly"""

        # super call needs to be last, because otherwise the new position of the lilypad is already used
        old_x = self.x
        super().update()

        # also update player position if on lilypad
        player = self.world.player
        if player.rect.colliderect(self.rect):
            # only if lilypad moved in this step
            if old_x != self.x:
                player.set_position((player.x + self.delta_x, player.y))
