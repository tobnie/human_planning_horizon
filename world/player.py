import os

import config
from world.dynamic_object import DynamicObject


class Player(DynamicObject):
    """
    Spawn a player
    """

    def __init__(self, world, start_position=(0, 0)):
        super().__init__(start_position[0], start_position[1], movement_bounds_x=(0, config.MONITOR_WIDTH_PX), movement_bounds_y=(0, config.MONITOR_HEIGHT_PX),
                         img_path=os.path.join(config.SPRITES_DIR, 'player.png'))
        self.world = world
        self.input_delta_x = 0
        self.input_delta_y = 0

    def set_position(self, pos):
        """Sets the current position of the player given as (x, y)-tuple."""
        self.rect.x = pos[0]
        self.rect.y = pos[1]

    def check_player_on_lilypad(self):
        """
        Checks if the player currently stands on a lilypad and adjusts the delta_x accordingly.
        """
        player_lilypads = self.world.check_player_on_lilypad()
        if player_lilypads:
            self.delta_x = player_lilypads[0].delta_x
        else:
            self.delta_x = 0

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The player is constrained by their movement boundaries."""
        self.check_player_on_lilypad()

        delta_x_tot = self.delta_x + self.input_delta_x
        delta_y_tot = self.delta_y + self.input_delta_y

        new_x = self.rect.x + config.STEP_SIZE * delta_x_tot
        new_y = self.rect.y + config.STEP_SIZE * delta_y_tot

        # x position
        if new_x < self.movement_bounds_x[0]:
            self.rect.x = 0
        elif new_x > self.movement_bounds_x[1] - self.rect.width:
            self.rect.x = self.movement_bounds_x[1] - self.rect.width
        else:
            self.rect.x = new_x

        # y position
        if new_y < self.movement_bounds_y[0]:
            self.rect.y = 0
        elif new_y > self.movement_bounds_y[1] - self.rect.height:
            self.rect.y = self.movement_bounds_y[1] - self.rect.height
        else:
            self.rect.y = new_y

    def print_information(self) -> None:
        print(f"Player Position (x, y) =  ({self.rect.x}, {self.rect.y})")
