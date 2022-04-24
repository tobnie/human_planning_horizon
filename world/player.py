import os

import numpy as np
import pygame

import config


class Player(pygame.sprite.Sprite):
    """
    Spawn a player
    """

    def __init__(self):
        self.delta_x = 0
        self.delta_y = 0

        pygame.sprite.Sprite.__init__(self)
        self.images = []
        for i in range(1, 5):
            img = pygame.image.load(os.path.join('sprites', 'player.png')).convert()
            img = pygame.transform.scale(img, 10 * np.array([13, 9]))
            self.images.append(img)
            self.image = self.images[0]
            self.rect = self.image.get_rect()
            self.rect.x = 0
            self.rect.y = 0

    def update(self):
        """Updates the players position by adding the current deltas to the current position."""
        self.rect.x = self.rect.x + config.STEP_SIZE * self.delta_x
        self.rect.y = self.rect.y + config.STEP_SIZE * self.delta_y
