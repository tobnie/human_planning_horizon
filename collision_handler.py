import pygame

from world.lane import Lane


def check_collision(sprite1: pygame.sprite.Sprite, sprite2: pygame.sprite.Sprite):
    """
    :return: True, if the rects of the two given sprites collide. False, otherwise.
    """

    return pygame.sprite.collide_rect(sprite1, sprite2)


def check_collision_group(sprite: pygame.sprite.Sprite, sprite_group: pygame.sprite.Group):
    """
    :return: True, if the given sprite collides with at least one sprite in the given sprite group. False, otherwise.
    """
    for sprite_g in sprite_group:
        if check_collision(sprite, sprite_g):
            return True
    return False





