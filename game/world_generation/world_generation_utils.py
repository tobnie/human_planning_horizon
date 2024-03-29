import json

from config import LEVELS_DIR
from game.world.world import World


def save_world(world: World, filename, path=LEVELS_DIR,):
    """Saves the given world as json at the given path."""

    # save world as json
    print("Saving world \'{}\'...".format(filename))
    with open(path + filename + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(world, ensure_ascii=False, indent=4))
    print("World \'{}\' saved.".format(filename))


def save_world_dict(world_dict, filename, path=LEVELS_DIR, verbose=False):
    """Saves the world (given as dict) as json at the given path."""

    # save world as json
    if verbose:
        print("Saving world \'{}\'...".format(filename))
    with open(path + filename + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(world_dict, ensure_ascii=False, indent=4))

    if verbose:
        print("World \'{}\' saved.".format(filename))
