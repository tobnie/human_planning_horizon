import os

import numpy as np
import pygame

import config


class Player(pygame.sprite.Sprite):
    """
    Spawn a player
    """

    def __init__(self, world):
        self.world = world

        self.delta_x = 0
        self.delta_y = 0

        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load(os.path.join('sprites', 'player.png')).convert()
        img = pygame.transform.scale(img, 10 * np.array([13, 9]))
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 0

    def update(self):
        """Updates the players position by adding the current deltas to the current position."""
        new_x = self.rect.x + config.STEP_SIZE * self.delta_x
        new_y = self.rect.y + config.STEP_SIZE * self.delta_y

        # x position
        if new_x < 0:
            self.rect.x = 0
        elif new_x > self.world.width - self.rect.width:
            self.rect.x = self.world.width - self.rect.width
        else:
            self.rect.x = self.rect.x + config.STEP_SIZE * self.delta_x

        # y position
        if new_y < 0:
            self.rect.y = 0
        elif new_y > self.world.height - self.rect.height:
            self.rect.y = self.world.height - self.rect.height
        else:
            self.rect.y = self.rect.y + config.STEP_SIZE * self.delta_y

    def print_information(self):
        print(f"Player Position (x, y) =  ({self.rect.x}, {self.rect.y})")
