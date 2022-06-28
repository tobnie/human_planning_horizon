from generation_config import GameDifficulty
from world_generation.world_generator import WorldGenerator

from tqdm import tqdm

""" Generates worlds and saves them to config.LEVELS_DIR """

n_worlds = 60
assert n_worlds % 3 == 0, "n_worlds must be divisible by 3"

world_generator = WorldGenerator()

for i in tqdm(range(n_worlds)):

    # generate new world
    world_name = "world_{}".format(i - (n_worlds // 3) * (i // (n_worlds // 3)))
    if i < n_worlds // 3:
        world = world_generator.generate_and_save_world(GameDifficulty.EASY, world_name=world_name)
    elif i < 2 * n_worlds // 3:
        world = world_generator.generate_and_save_world(GameDifficulty.NORMAL, world_name=world_name)
    else:
        world = world_generator.generate_and_save_world(GameDifficulty.HARD, world_name=world_name)
