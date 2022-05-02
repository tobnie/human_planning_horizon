import os
import random
import string

import pygame

import colors
import config
import event_handler
from world.world import World

'''
Variables
'''


class Game:

    def __init__(self):
        """
        Sets up the game by initializing PyGame.
        """
        # game clock
        self.clock = pygame.time.Clock()
        self.game_clock = 0
        self.display_debug_information = False
        self.running = True

        # collision counter
        self.vehicle_collision = False
        self.water_collision = False

        # set screen information
        pygame.init()
        self.screen = pygame.display.set_mode((config.MONITOR_WIDTH_PX, config.MONITOR_HEIGHT_PX), pygame.FULLSCREEN)

        self.font_color = colors.WHITE
        self.font_size = config.font_size
        self.font = pygame.freetype.SysFont(name="freesansbold", size=self.font_size)

        self.world = World(config.MONITOR_WIDTH_PX, config.MONITOR_HEIGHT_PX)
        self.screen.fill(colors.BLACK)

    def run(self):
        """
        Main Loop
        """

        while self.running:

            # event handling
            event_handler.handle_events(self)

            # spawning street
            # TODO outsource to spawn_handler or similar?
            if self.game_clock % 2 == 0:  # TODO spawns every 10000th tick now
                for lane in self.world.directed_lanes:
                    lane.spawn_entity()
                self.world.update()

                # check collision
                self.check_collision()

            # draw objects
            self.render()

            # tick game
            self.clock.tick(config.FPS)
            self.game_clock += 1

    def check_collision(self):
        self.vehicle_collision = True if self.world.check_vehicle_collision() else False
        self.water_collision = True if self.world.check_water_collision() else False

    def render(self):
        self.world.draw(self.screen)

        if self.display_debug_information:
            self.render_debug_information()

        pygame.display.flip()

    def render_debug_information(self):
        debug_info = self.debug_information()

        for i, debug_line in enumerate(debug_info):
            text_surface, rect = self.font.render(debug_line, self.font_color)
            self.screen.blit(text_surface, (0, i * self.font_size))

    def debug_information(self) -> string:
        debug_information = [f"Player Position = ({self.world.player.rect.x}, {self.world.player.rect.y})",
                             f"Player Delta = ({self.world.player.delta_x}, {self.world.player.delta_y})",
                             f"Vehicle Collision = {self.vehicle_collision}",
                             f"Water Collision = {self.water_collision}"]
        return debug_information
