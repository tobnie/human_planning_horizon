import pygame

import config


class Field(pygame.sprite.Sprite):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x * config.FIELD_WIDTH, y * config.FIELD_HEIGHT, config.FIELD_WIDTH, config.FIELD_HEIGHT)

