import os
import string

import numpy as np
import pygame

import config


class GameObject(pygame.sprite.Sprite):

    def __init__(self, x: int = 0, y: int = 0, width: int = 0.0, height: int = 0.0, img_path: string = None):
        super().__init__()

        # sprite image
        if img_path:
            img = pygame.image.load(img_path).convert_alpha()
            img = pygame.transform.scale(img, np.array(
                [config.MONITOR_HEIGHT_PX / config.N_LANES + 1, config.MONITOR_HEIGHT_PX / config.N_LANES + 1]))  # TODO scaling
            self.image = img

            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
        else:
            self.rect = pygame.Rect(x, y, width, height)


class StaticObject(GameObject):

    def __init__(self, x: int = 0, y: int = 0, img_path: string = None):
        super().__init__(x, y, img_path=img_path)


class DynamicObject(GameObject):

    def __init__(self, x: int = 0, y: int = 0, movement_bounds_x: (int, int) = (0, config.MONITOR_WIDTH_PX),
                 movement_bounds_y: (int, int) = (0, config.MONITOR_HEIGHT_PX), img_path: string = None):
        super().__init__(x, y, img_path=img_path)

        # dynamics
        self.movement_bounds_x = movement_bounds_x
        self.movement_bounds_y = movement_bounds_y
        self.delta_x = 0
        self.delta_y = 0

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The sprite dies if it moves outside of the movement boundaries."""
        new_x = self.rect.x + config.STEP_SIZE * self.delta_x
        new_y = self.rect.y + config.STEP_SIZE * self.delta_y

        # x position
        if new_x < self.movement_bounds_x[0] or new_x > self.movement_bounds_x[1] - self.rect.width:
            self.kill()
        else:
            self.rect.x = self.rect.x + config.STEP_SIZE * self.delta_x

        # y position
        if new_y < self.movement_bounds_y[0] or new_y > self.movement_bounds_y[1] - self.rect.height:
            self.kill()
        else:
            self.rect.y = self.rect.y + config.STEP_SIZE * self.delta_y


class Vehicle(DynamicObject):

    def __init__(self, x: int = 0, y: int = 0,
                 movement_bounds_x: (int, int) = (-config.OBJECT_WIDTH, config.MONITOR_WIDTH_PX + config.OBJECT_WIDTH),
                 movement_bounds_y: (int, int) = (0, config.MONITOR_HEIGHT_PX)):
        super().__init__(x, y, movement_bounds_x, movement_bounds_y, os.path.join(config.SPRITES_DIR, "car.png"))
        # TODO ?


class LilyPad(DynamicObject):
    def __init__(self, x: int = 0, y: int = 0,
                 movement_bounds_x: (int, int) = (-config.OBJECT_WIDTH, config.MONITOR_WIDTH_PX + config.OBJECT_WIDTH),
                 movement_bounds_y: (int, int) = (0, config.MONITOR_HEIGHT_PX)):
        super().__init__(x, y, movement_bounds_x, movement_bounds_y, os.path.join(config.SPRITES_DIR, "none.png"))
        # TODO ?
