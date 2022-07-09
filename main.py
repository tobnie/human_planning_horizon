import pygame
pygame.init()

import config
from world_generation.generation_config import GameDifficulty
from game import Game

screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)
game = Game(GameDifficulty.EASY, "world_0", screen=screen)
game.run()
