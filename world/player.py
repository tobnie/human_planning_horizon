import os

import config
from world.game_object import DynamicObject


class Player(DynamicObject):
    """
    Spawn a player
    """

    def __init__(self, world, start_position=(0, config.N_FIELDS_PER_LANE // 2 + 1)):
        super().__init__(start_position[0], start_position[1], config.FIELD_WIDTH, config.FIELD_HEIGHT,
                         movement_bounds_x=(0, config.N_FIELDS_PER_LANE), movement_bounds_y=(0, config.N_LANES),
                         img_path=os.path.join(config.SPRITES_DIR, 'player.png'))
        self.world = world

    def check_player_on_lilypad(self):
        """
        Checks if the player currently stands on a lilypad and adjusts the delta_x accordingly.
        """
        player_lilypads = self.world.check_player_on_lilypad()
        if player_lilypads:
            # TODO move with lilypad
            self.delta_x = player_lilypads[0].delta_x

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The player is constrained by their movement boundaries."""

        # self.check_player_on_lilypad() # TODO

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

    def print_information(self) -> None:
        print(f"Player Position (x, y) =  ({self.rect.x}, {self.rect.y})")
