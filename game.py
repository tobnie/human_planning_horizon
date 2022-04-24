import pygame

import config
import event_handler
from world.player import Player

'''
Variables
'''

fps = 40  # frame rate
screen = pygame.display.set_mode([config.MONITOR_WIDTH_PX, config.MONITOR_HEIGHT_PX])

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

        pygame.init()
        screen.fill(BLACK)

        self.player = Player()  # spawn player
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)

    def run(self):
        """
        Main Loop
        """

        while self.running:
            event_handler.handle_events(self)
            self.player.update()

            self.player_list.draw(screen)
            pygame.display.flip()
            self.clock.tick(fps)

    def print_player_information(self):
        print(f"Player Position (x, y) =  ({self.player.rect.x}, {self.player.rect.y})")
