import pygame

import config


class Field(pygame.sprite.Sprite):

    def __init__(self, x, y, color=None, img_path=None):
        super().__init__()
        self.x = x
        self.y = y
        self.color = color

        # sprite image
        if img_path:
            img = pygame.image.load(img_path).convert_alpha()
            img = pygame.transform.scale(img, [config.FIELD_WIDTH, config.FIELD_HEIGHT])
            self.base_img = img
            self.image = img
            self.rect = self.image.get_rect()
            self.rect.x = x * config.FIELD_WIDTH
            self.rect.y = y * config.FIELD_HEIGHT
        else:
            self.rect = pygame.Rect(x * config.FIELD_WIDTH, y * config.FIELD_HEIGHT, config.FIELD_WIDTH, config.FIELD_HEIGHT)
            self.image = None

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, (self.rect.x, self.rect.y))