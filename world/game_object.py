import os
import string
from abc import ABC

import pygame

import colors
import config


class GameObject(pygame.sprite.Sprite, ABC):
    """An abstract class for game objects"""

    def __init__(self, world, x: int = 0, y: int = 0, width: int = 1, height: int = 1, img_path: string = None):
        super().__init__()

        # assign object id
        self.id = id(self)
        self.world = world
        self.rotatable = True

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
        self.x: int = x
        self.y: int = y
        self.set_position((x, y))

    def set_position(self, pos: (int, int)):
        """Sets the current position of the player given as (x, y)-tuple."""

        self.x = pos[0]
        self.y = pos[1]
        self.rect.x = self.x * config.FIELD_WIDTH
        self.rect.y = self.y * config.FIELD_HEIGHT


class StaticObject(GameObject):
    """A class for static game objects, i.e. they have no ability to move."""

    def __init__(self, world, x: int = 0, y: int = 0, width: int = 1, height: int = 1, img_path: string = None):
        super().__init__(world, x, y, width, height, img_path=img_path)


class DynamicObject(GameObject):
    """A class for dynamic game objects, which are moving with a specified delta in either x- or y-direction."""

    def __init__(self, world, x: int = 0, y: int = 0, width: int = 1, height: int = 1,
                 img_path: string = None, delta_x: int = 0,
                 delta_y: int = 0,
                 movement_bounds_x: (int, int) = (-1, config.N_FIELDS_PER_LANE),
                 movement_bounds_y: (int, int) = (-1, config.N_LANES)):
        super().__init__(world, x, y, width, height, img_path=img_path)

        # dynamics
        self.movement_bounds_x = movement_bounds_x
        self.movement_bounds_y = movement_bounds_y
        self.delta_x: int = delta_x
        self.delta_y: int = delta_y

    def set_rotated_sprite_img(self):
        """Sets the sprite image for the respective current direction of movement."""
        if self.delta_x > 0:
            self.image = pygame.transform.rotate(self.base_img, 270)
        if self.delta_y > 0:
            self.image = pygame.transform.rotate(self.base_img, 180)
        if self.delta_x < 0:
            self.image = pygame.transform.rotate(self.base_img, 90)
        if self.delta_y < 0:
            self.image = pygame.transform.rotate(self.base_img, 0)

        self.image = pygame.transform.scale(self.image, [self.rect.width, self.rect.height])
        self.image.set_colorkey(colors.WHITE)

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The sprite dies if it moves outside of the movement boundaries."""

        if self.rotatable:
            self.set_rotated_sprite_img()

        new_x = self.x + abs(self.delta_x) / self.delta_x
        new_y = self.y + self.delta_y

        # x position
        if new_x < self.movement_bounds_x[0] or new_x > self.movement_bounds_x[1]:
            self.kill()

        # y position
        if new_y < self.movement_bounds_y[0] or new_y > self.movement_bounds_y[1]:
            self.kill()

        # TODO update rect separately for pseudo-continuous display
        # only update rect coordinates for pseudo-continuous display
        sign = -1 if self.delta_x < 0 else (1 if self.delta_x > 0 else 0)
        self.rect.x = self.rect.x + sign / config.OBSTACLE_SPAWN_RATE

        self.set_position((new_x, new_y))


class Vehicle(DynamicObject):
    """Vehicles are moving on the streets with specific properties given by the lane they are on (passed to the
    vehicle constructor upon spawning in the lane)"""

    def __init__(self, world, x: int = 0, y: int = 0,
                 width: int = 1, height: int = 1,
                 delta_x: int = 0,
                 delta_y: int = 0):
        super().__init__(world, x, y, width, height, os.path.join(config.SPRITES_DIR, "car.jpg"), delta_x,
                         delta_y)


class LilyPad(DynamicObject):
    """Lilypads are moving on water with specific properties given by the lane they are on (passed to the
        lilypad constructor upon spawning in the lane)"""

    def __init__(self, world, x: int = 0, y: int = 0,
                 width: int = 1, height: int = 1,
                 delta_x: int = 0,
                 delta_y: int = 0):
        if width == 1:
            img_file = config.LILYPAD_FILE
        else:
            img_file = config.LOG_FILE
        super().__init__(world, x, y, width, height, os.path.join(config.SPRITES_DIR, img_file), delta_x,
                         delta_y)
        self.rotatable = False

    def update(self) -> None:
        """Calls the super method as usual but also checks if the player is on the lilypad and if so,
        the player's position is also updated accordingly"""

        # also update player position if on lilypad
        player = self.world.player
        if player.rect.colliderect(self.rect):
            player.set_position((player.x + abs(self.delta_x) / self.delta_x, player.y))

        # super call needs to be last, because otherwise the new position of the lilypad is already used
        super().update()
