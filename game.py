import pygame

from display_debug_information import TextDisplayer

import colors
import config
import event_handler
from world.world import World, WorldStatus


class Game:

    def __init__(self, world_name: str = None):
        """
        Sets up the game by initializing PyGame.
        """
        # game clock
        self.clock = pygame.time.Clock()

        self.display_debug_information_player = False
        self.display_debug_information_objects = False
        self.display_debug_information_lanes = False
        self.running = True
        self.pause = False

        self.world_status = WorldStatus.RUNNING
        self.world_name = world_name

        # collision counter
        self.vehicle_collision = False
        self.water_collision = False

        # set screen information
        self.screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)

        if world_name:
            self.world = World(self, world_name=world_name)
        else:
            self.world = World(self, config.N_FIELDS_PER_LANE, config.N_LANES)
        self.screen.fill(colors.BLACK)

        self.text_displayer = TextDisplayer(self)

    def run_pause(self):
        event_handler.handle_events(self)

    def run_normal(self):
        self.world_status = self.world.update()

        # draw objects
        self.render()

        # tick game
        self.clock.tick(config.FPS)
        self.world.game_clock += 1

    def run(self):
        """
        Main Loop
        """

        while self.running:

            # run next game step
            if self.pause:
                self.run_pause()
            else:
                self.run_normal()

            if self.world_status == WorldStatus.WON:
                # game won
                self.start_world(self.world_name)
            if self.world_status == WorldStatus.LOST:
                # game lost
                self.start_world(self.world_name)

    def start_world(self, world_name):
        self.world = World(self, world_name=world_name)
        self.text_displayer = TextDisplayer(self)

    def render(self):
        """Renders the whole game."""
        self.world.draw(self.screen)

        self.text_displayer.display_debug_information()

        pygame.display.flip()
