import json
from world.world import World


def save_world(world, path, filename):
    """Saves the given world as json at the given path."""

    # save world as json
    print("Saving world \'{}\'...".format(filename))
    with open(path + filename + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(world, ensure_ascii=False, indent=4))
    print("World \'{}\' saved.".format(filename))


def load_world(path, filename):
    """Loads and returns the world from the json at the given path."""

    print("Loading world from {}...".format(path))
    world = World(path=path + filename + '.json')
    print("World \'{}\' loaded.".format(filename))

    return world
