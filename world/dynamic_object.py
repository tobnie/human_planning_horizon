import os
import string

import numpy as np
import pygame

import config


class GameObject(pygame.sprite.Sprite):

    def __init__(self, x: float = 0, y: float = 0, width: float = 0.0, height: float = 0.0, img_path: string = None):
        super().__init__()

        # internal position (float)
        self.x: float = x
        self.y: float = y

        # sprite image
        if img_path:
            img = pygame.image.load(img_path).convert_alpha()
            print(f"[width, height]={[width, height]}")
            img = pygame.transform.scale(img, [width, height])
            # [config.MONITOR_HEIGHT_PX / config.N_LANES + 1, config.MONITOR_HEIGHT_PX / config.N_LANES + 1]))  # TODO scaling
            self.image = img

            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
        else:
            self.rect = pygame.Rect(x, y, width, height)


class StaticObject(GameObject):

    def __init__(self, x: float = 0, y: float = 0, width: float = 0, height: float = 0, img_path: string = None):
        super().__init__(x, y, width, height, img_path=img_path)


class DynamicObject(GameObject):

    def __init__(self, x: float = 0, y: float = 0, width: float = 0, height: float = 0,
                 movement_bounds_x: (float, float) = (0, config.MONITOR_WIDTH_PX),
                 movement_bounds_y: (float, float) = (0, config.MONITOR_HEIGHT_PX), img_path: string = None, delta_x: float = 0,
                 delta_y: float = 0):
        super().__init__(x, y, width, height, img_path=img_path)

        # dynamics
        self.movement_bounds_x = movement_bounds_x
        self.movement_bounds_y = movement_bounds_y
        self.delta_x: float = delta_x
        self.delta_y: float = delta_y

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The sprite dies if it moves outside of the movement boundaries."""
        new_x = self.rect.x + self.delta_x
        new_y = self.rect.y + self.delta_y

        # x position
        if new_x < self.movement_bounds_x[0] or new_x > self.movement_bounds_x[1] - self.rect.width:
            self.kill()
        else:
            self.rect.x = self.rect.x + self.delta_x

        # y position
        if new_y < self.movement_bounds_y[0] or new_y > self.movement_bounds_y[1] - self.rect.height:
            self.kill()
        else:
            self.rect.y = self.rect.y + self.delta_y


class Vehicle(DynamicObject):

    def __init__(self, x: float = 0, y: float = 0,
                 width: float = 0, height: float = 0,
                 movement_bounds_x: (float, float) = (-config.OBSTACLE_WIDTH, config.MONITOR_WIDTH_PX + config.OBSTACLE_WIDTH),
                 movement_bounds_y: (float, float) = (0, config.MONITOR_HEIGHT_PX), delta_x: float = 0,
                 delta_y: float = 0):
        super().__init__(x, y, width, height, movement_bounds_x, movement_bounds_y, os.path.join(config.SPRITES_DIR, "car.png"), delta_x,
                         delta_y)


class LilyPad(DynamicObject):
    def __init__(self, x: float = 0, y: float = 0,
                 width: float = 0, height: float = 0,
                 movement_bounds_x: (float, float) = (-config.OBSTACLE_WIDTH, config.MONITOR_WIDTH_PX + config.OBSTACLE_WIDTH),
                 movement_bounds_y: (float, float) = (0, config.MONITOR_HEIGHT_PX), delta_x: float = 0,
                 delta_y: float = 0):
        super().__init__(x, y, width, height, movement_bounds_x, movement_bounds_y, os.path.join(config.SPRITES_DIR, "none.png"), delta_x,
                         delta_y)
