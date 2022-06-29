import pygame

from world_generation.generation_config import GameDifficulty

pygame.init()

from game import Game

game = Game(GameDifficulty.EASY, "easy/world_0")
game.run()
