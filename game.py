import os
import random

import pygame

import colors
import config
import event_handler
from world.player import Player
from world.world import World

'''
Variables
'''


class Game:

    def __init__(self):
        """
        Sets up the game by initializing PyGame.
        """
        self.clock = pygame.time.Clock()
        self.game_clock = 0
        self.running = True

        # set screen information
        pygame.init()
        self.screen = pygame.display.set_mode((config.MONITOR_WIDTH_PX, config.MONITOR_HEIGHT_PX), pygame.FULLSCREEN)
        self.world = World(config.MONITOR_WIDTH_PX, config.MONITOR_HEIGHT_PX)
        self.screen.fill(colors.BLACK)

    def run(self):
        """
        Main Loop
        """

        while self.running:
            # event handling
            event_handler.handle_events(self)
            self.world.update()

            # spawning street
            # TODO outsource to spawn_handler or similar?
            if self.game_clock % 100 == 0:  # TODO spawns every 10000th tick now
                for lane in self.world.get_directed_lanes():
                    if random.choice([True, False]):
                        lane.spawn_entity()

            # draw objects
            self.render()

            # tick game
            self.clock.tick(config.FPS)
            self.game_clock += 1

    def render(self):
        self.world.draw(self.screen)
