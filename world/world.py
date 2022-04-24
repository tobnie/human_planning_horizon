import pygame

from world.player import Player


class World:

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.player = Player(self)  # spawn player
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)

        self.lanes = []  # TODO generate lanes
