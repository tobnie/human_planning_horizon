import os

import config
from world.dynamic_object import DynamicObject


class Player(DynamicObject):
    """
    Spawn a player
    """

    def __init__(self, world):
        super().__init__(0, 0, movement_bounds_x=(0, config.MONITOR_WIDTH_PX), movement_bounds_y=(0, config.MONITOR_HEIGHT_PX),
                         img_path=os.path.join(config.SPRITES_DIR, 'player.png'))
        self.world = world

    def update(self) -> None:
        """Updates the object's position by adding the current deltas to the current position.
        The player is constrained by their movement boundaries."""
        new_x = self.rect.x + config.STEP_SIZE * self.delta_x
        new_y = self.rect.y + config.STEP_SIZE * self.delta_y

        # x position
        if new_x < self.movement_bounds_x[0]:
            self.rect.x = 0
        elif new_x > self.movement_bounds_x[1] - self.rect.width:
            self.rect.x = self.movement_bounds_x[1] - self.rect.width
        else:
            self.rect.x = self.rect.x + config.STEP_SIZE * self.delta_x

        # y position
        if new_y < self.movement_bounds_y[0]:
            self.rect.y = 0
        elif new_y > self.movement_bounds_y[1] - self.rect.height:
            self.rect.y = self.movement_bounds_y[1] - self.rect.height
        else:
            self.rect.y = self.rect.y + config.STEP_SIZE * self.delta_y

    def print_information(self) -> None:
        print(f"Player Position (x, y) =  ({self.rect.x}, {self.rect.y})")
