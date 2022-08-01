import pygame
pygame.init()

import config
from world_generation.generation_config import GameDifficulty
from game import Game

difficulty = GameDifficulty.HARD
world_number = 0

screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)
game = Game(difficulty=difficulty, eye_tracker=None, world_name="world_{}".format(world_number), screen=screen)
game.pre_run()
game.run()
