import pygame

from display_debug_information import TextDisplayer

import colors
import config
import event_handler
from world.world import World

'''
Variables
'''


class Game:

    def __init__(self, world_name: str = None):
        """
        Sets up the game by initializing PyGame.
        """
        # game clock
        self.clock = pygame.time.Clock()
        self.game_clock = 0
        self.display_debug_information_player = False
        self.display_debug_information_objects = False
        self.running = True
        self.pause = False

        self.game_won = False

        # collision counter
        self.vehicle_collision = False
        self.water_collision = False

        # set screen information
        self.screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)

        if world_name:
            self.world = World(config.N_FIELDS_PER_LANE, config.N_LANES, world_name=world_name)
        else:
            self.world = World(config.N_FIELDS_PER_LANE, config.N_LANES)
        self.screen.fill(colors.BLACK)

        self.text_displayer = TextDisplayer(self)

    def run_pause(self):
        event_handler.handle_events(self)

    def run_normal(self):
        # event handling
        if self.game_clock % 4 == 0:
            event_handler.handle_events(self)
            self.world.player_update()

        # spawning street
        # TODO outsource to spawn_handler or similar?
        if self.game_clock % 50 == 0:  # TODO
            for lane in self.world.directed_lanes:
                lane.spawn_entity()
            self.world.update()

        # check if player is dead and end game
        if self.world.player.check_status():
            pass  # TODO

        # draw objects
        self.render()

        # tick game
        self.clock.tick(config.FPS)
        self.game_clock += 1

    def run(self):
        """
        Main Loop
        """

        while self.running:

            if self.pause:
                self.run_pause()
            else:
                self.run_normal()

    def check_game_won(self):
        """
        Checks if the game has been won.
        """
        # TODO extend to target positions
        if self.world.player.y == 0:
            self.game_won = True

    def render(self):
        """Renders the whole game."""
        self.world.draw(self.screen)

        self.text_displayer.display_debug_information()

        pygame.display.flip()
