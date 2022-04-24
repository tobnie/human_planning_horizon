import os

import pygame

import config
import event_handler
from world.player import Player
from world.world import World

'''
Variables
'''

fps = 40  # frame rate

BLUE = (25, 25, 200)
BLACK = (23, 23, 23)
WHITE = (254, 254, 254)
ALPHA = (0, 255, 0)


class Game:

    def __init__(self):
        """
        Sets up the game by initializing PyGame.
        """
        self.clock = pygame.time.Clock()
        self.running = True

        # get monitor size information
        os.environ['SDL_VIDEO_CENTERED'] = '1'  # You have to call this before pygame.init()
        pygame.init()
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h

        self.screen = pygame.display.set_mode([screen_width, screen_height])
        self.world = World(screen_width, screen_height)


        self.screen.fill(BLACK)

    def run(self):
        """
        Main Loop
        """

        while self.running:
            # event handling
            event_handler.handle_events(self)
            self.world.player.update()

            # draw objects
            self.screen.fill(BLACK)
            self.world.player_list.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(fps)
