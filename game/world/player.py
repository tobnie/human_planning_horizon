import os
from enum import Enum

import pygame

from game import collision_handler
import colors
import config
from game.world.game_object import DynamicObject
from game.world.lane import WaterLane, DirectedLane


class PlayerAction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    NONE = 4


class Player(DynamicObject):
    """
    Player Class
    """

    def __init__(self, world, start_position=(0, (config.N_FIELDS_PER_LANE // 2 + 1) * config.FIELD_WIDTH)):

        super().__init__(world, start_position[0], start_position[1], 0, config.PLAYER_WIDTH_TO_FIELD_WIDTH_RATIO,
                         config.PLAYER_HEIGHT_TO_FIELD_HEIGHT_RATIO,
                         movement_bounds_x=config.PLAYER_MOVEMENT_BOUNDS_X, movement_bounds_y=config.PLAYER_MOVEMENT_BOUNDS_Y,
                         img_path=os.path.join(config.SPRITES_DIR, '../sprites/player.png'))

        self.delta_x = 0
        self.delta_y = 0
        self.is_dead = False
        self.current_action = PlayerAction.NONE
        self.highest_visited_lane = 0

    def set_position(self, pos: (float, float)):
        super().set_position(pos)
        self.highest_visited_lane = max(self.highest_visited_lane, self.y // config.FIELD_HEIGHT)

    def get_action(self) -> PlayerAction:
        """ Returns the currently executed action of the player. """
        return self.current_action

    def set_action(self, action: PlayerAction) -> None:
        """Sets the current action of the player and also sets their delta values accordingly."""
        self.current_action = action

        if action == PlayerAction.RIGHT:
            self.delta_x = 1
        elif action == PlayerAction.LEFT:
            self.delta_x = -1
        elif action == PlayerAction.UP:
            self.delta_y = -1
        elif action == PlayerAction.DOWN:
            self.delta_y = 1
        else:
            self.delta_x = 0
            self.delta_y = 0

    def check_player_on_lilypad(self):
        """Returns True if the player is in a WaterLane on a Lilypad. Otherwise, False."""
        for lane in self.world.lanes:
            if isinstance(lane, WaterLane):
                for sprite in lane.non_player_sprites:
                    if self.rect.colliderect(sprite.rect):
                        if sprite.rect.x < self.rect.centerx < sprite.rect.x + sprite.rect.width:
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
        on_water = self.y // config.FIELD_HEIGHT in water_rows

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

        self.set_rotated_sprite_img()

        new_x = self.x + self.delta_x * config.FIELD_WIDTH
        new_y = self.y + self.delta_y * config.FIELD_HEIGHT

        # reset delta_x and delta_y
        self.delta_x = 0
        self.delta_y = 0

        # check bounds in x-direction
        if self.y != 0:
            if new_x <= self.movement_bounds_x[0]:
                self.kill()
            elif new_x >= self.movement_bounds_x[1]:
                self.kill()
            else:
                new_x = new_x
        else:
            new_x = self.x

        # check bounds in y-direction
        if new_y <= self.movement_bounds_y[0] + config.FIELD_HEIGHT:
            # only update y if the player will end on the target position
            target_x = self.world.finish_lanes.sprites()[0].target_position * config.FIELD_WIDTH
            new_center_x = new_x + self.rect.width / 2
            if target_x <= new_center_x <= target_x + config.FIELD_WIDTH:
                margin_y = (1 - config.PLAYER_HEIGHT_TO_FIELD_HEIGHT_RATIO) / 2 * config.FIELD_HEIGHT
                new_y = self.movement_bounds_y[0] + margin_y
            else:
                new_y = self.y
        elif new_y >= self.movement_bounds_y[1]:
            margin_y = (1 - config.PLAYER_HEIGHT_TO_FIELD_HEIGHT_RATIO) / 2 * config.FIELD_HEIGHT
            new_y = self.movement_bounds_y[1] + margin_y
        else:
            new_y = new_y

        self.set_position((new_x, new_y))

    def set_rotated_sprite_img(self):
        """Sets the sprite image for the respective current direction of movement."""
        if self.delta_x > 0:
            self.image = pygame.transform.rotate(self.base_img, 270)
        if self.delta_y > 0:
            self.image = pygame.transform.rotate(self.base_img, 180)
        if self.delta_x < 0:
            self.image = pygame.transform.rotate(self.base_img, 90)
        if self.delta_y < 0:
            self.image = pygame.transform.rotate(self.base_img, 0)

        self.image = pygame.transform.scale(self.image, [self.rect.width, self.rect.height])
        self.image.set_colorkey(colors.WHITE)

    def kill(self) -> None:
        """ Kills the player. """
        self.is_dead = True
        super().kill()

    def print_information(self) -> None:
        print(f"Player Position (x, y) =  ({self.rect.x}, {self.rect.y})")
