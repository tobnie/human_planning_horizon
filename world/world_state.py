import numpy as np

from world.game_object import Vehicle, LilyPad
from world.lane import DirectedLane
from world.player import Player


class WorldState:
    OBJECT_TYPE_TO_INT = {
        Player: 0,
        Vehicle: 1,
        LilyPad: 2
    }

    def __init__(self, world):
        self.world = world
        self.object_arr = self._get_object_array()

    def _get_object_array(self):
        """ Returns an array representing all objects in the world state in the following form:
         [ObjectType, x, y, width]
         """

        # collect objects
        objects = []

        # add player
        # TODO save x (discrete) or rect.x (continuous)
        player_list = [self.OBJECT_TYPE_TO_INT[self.world.player.__class__], self.world.player.x, self.world.player.y,
                       self.world.player.width]
        objects.append(player_list)

        # add objects
        for lane in self.world.directed_lanes:
            if lane is isinstance(lane, DirectedLane):
                for obj in lane.non_player_sprites.sprites():
                    # TODO save x (discrete) or rect.x (continuous)
                    object_list = [self.OBJECT_TYPE_TO_INT[obj.__class__], obj.x, obj.y, obj.width]
                    objects.append(object_list)

        return np.asarray(objects)

    def save_to_file(self, filename):
        """Saves the world state (all objects with corresponding location and width) to a .npz-file."""
        np.savez_compressed(filename+".npz", self.object_arr)
