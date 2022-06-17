import json

import pygame
pygame.init()

import config
from world.world import World


def save_world(world, path, filename):
    """Saves the given world as json at the given path."""

    # save world as json
    with open(path + filename + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(world, ensure_ascii=False, indent=4))


def load_world(path):
    """Loads and returns the world from the json at the given path."""
    # TODO
    pass


screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)
world = World(config.N_FIELDS_PER_LANE, config.N_LANES)
save_world(world, '../levels/', 'test_world')