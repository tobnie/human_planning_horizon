from world_generation import world_generation_utils
from generation_config import GameDifficulty
from world_generation import WorldGenerator

from tqdm import tqdm

""" Generates worlds and saves them to config.LEVELS_DIR """

n_worlds = 300
assert n_worlds % 3 == 0, "n_worlds must be divisible by 3"

world_generator = WorldGenerator()

for i in tqdm(range(n_worlds)):

    # generate new world
    world_name = "world_{}".format(i - 100 * (i // 100))
    if i < n_worlds // 3:
        world = world_generator.generate_and_save_world(GameDifficulty.EASY, world_name=world_name)
    elif i < 2 * n_worlds // 3:
        world = world_generator.generate_and_save_world(GameDifficulty.NORMAL, world_name=world_name)
    else:
        world = world_generator.generate_and_save_world(GameDifficulty.HARD, world_name=world_name)
