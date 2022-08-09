import numpy as np

from game.world.game_object import Vehicle, LilyPad
from game.world.lane import DirectedLane
from game.world.player import Player


class WorldState:
    OBJECT_TYPE_TO_INT = {
        Player: 0,
        Vehicle: 1,
        LilyPad: 2
    }

    def __init__(self, world):
        self.world = world
        self.object_arr = self._get_object_array()

    def asarray(self):
        return self.object_arr

    def _get_object_array(self):
        """ Returns an array representing all objects in the world state in the following form:
         [ObjectType, x, y, width]
         """

        # collect objects
        game_objects = []

        # add player
        player_list = [self.OBJECT_TYPE_TO_INT[self.world.player.__class__], self.world.player.rect.x, self.world.player.rect.y,
                       self.world.player.rect.width]
        game_objects.append(player_list)

        # add objects
        for lane in self.world.directed_lanes:
            if isinstance(lane, DirectedLane):
                for obj in lane.non_player_sprites.sprites():
                    object_list = [self.OBJECT_TYPE_TO_INT[obj.__class__], obj.rect.x, obj.rect.y, obj.rect.width]
                    game_objects.append(object_list)

        return np.asarray(game_objects)
