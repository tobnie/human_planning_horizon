import os

import collision_handler
import config
from world.game_object import DynamicObject
from world.lane import WaterLane, DirectedLane


class Player(DynamicObject):
    """
    Spawn a player
    """

    def __init__(self, world, start_position=(0, config.N_FIELDS_PER_LANE // 2 + 1)):
        super().__init__(world, start_position[0], start_position[1], 1, 1,
                         movement_bounds_x=(0, config.N_FIELDS_PER_LANE-1), movement_bounds_y=(0, config.N_LANES-1),
                         img_path=os.path.join(config.SPRITES_DIR, 'player.png'))
        self.is_dead = False

    def check_player_on_lilypad(self):
        """Returns True if the player is in a WaterLane on a Lilypad. Otherwise, False."""
        for lane in self.world.lanes:
            if isinstance(lane, WaterLane):
                for sprite in lane.non_player_sprites:
                    if self.rect.colliderect(sprite.rect):
                        return True
        return False

    def check_vehicle_collision(self):
        """
        :return: True, if the player collides with a vehicle. False, otherwise
        """
        for street_lane in self.world.street_lanes:
            if isinstance(street_lane, DirectedLane) and collision_handler.check_collision_group(self,
                                                                                                 street_lane.non_player_sprites):
                return True

        return False

    def check_water_collision(self):
        """
        :return: True, if the player does not collide with a lilypad and thus collides with the water. Otherwise False.
        """
        # check collision with water
        water_rows = list(map(lambda lane: lane.row, self.world.water_lanes.sprites()))
        on_water = self.y in water_rows

        # return False if player is not on a water lane
        if not on_water:
            return False

        # check collision with lilypad
        return not self.check_player_on_lilypad()

    def check_status(self):
        """Sets the player to dead, if the player collides with a Vehicle sprite or not a
        LilyPad sprite while being in a water lane"""
        if self.check_vehicle_collision() or self.check_water_collision():
            self.world.player.is_dead = True

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The player is constrained by their movement boundaries."""

        if self.world.game_clock % config.PLAYER_UPDATE_RATE != 0:
            return

        self.set_rotated_sprite_img()

        new_x = self.x + self.delta_x
        new_y = self.y + self.delta_y

        # x position
        if new_x < self.movement_bounds_x[0]:
            new_x = self.movement_bounds_x[0]
        elif new_x > self.movement_bounds_x[1]:
            new_x = self.movement_bounds_x[1]
        else:
            new_x = new_x

        # y position
        if new_y < self.movement_bounds_y[0]:
            new_y = self.movement_bounds_y[0]
        elif new_y > self.movement_bounds_y[1]:
            new_y = self.movement_bounds_y[1]
        else:
            new_y = new_y

        self.set_position((new_x, new_y))

        # check if dead
        self.check_status()

    def print_information(self) -> None:
        print(f"Player Position (x, y) =  ({self.rect.x}, {self.rect.y})")
