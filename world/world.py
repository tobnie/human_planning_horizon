import random

import numpy as np
import pygame
from pygame.sprite import spritecollide

import collision_handler
import colors
import config
from world.lane import StartLane, StreetLane, WaterLane, FinishLane, Lane, LaneDirection, DirectedLane
from world.player import Player


class World:

    def __init__(self, width, height):
        # game boundaries
        self.width = width
        self.height = height

        # lanes
        self.lanes = pygame.sprite.Group()
        self.directed_lanes = pygame.sprite.Group()
        self.starting_lanes = pygame.sprite.Group()
        self.street_lanes = pygame.sprite.Group()
        self.middle_lanes = pygame.sprite.Group()
        self.water_lanes = pygame.sprite.Group()
        self.finish_lanes = pygame.sprite.Group()
        self._generate_lanes()

        # player
        self.player = Player(self)  # spawn player
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)

        # start position
        starting_lane: StartLane = self.starting_lanes.sprites()[0]
        player_start_x = starting_lane.starting_position[0]
        player_start_y = starting_lane.starting_position[1]
        self.player.set_position((player_start_x, player_start_y))

    def draw(self, screen) -> None:
        screen.fill(colors.BLACK)

        # draw lanes
        self.draw_lanes(screen)

        # draw player
        self.player_list.draw(screen)

    def player_update(self):
        """
        Updates the player.
        """
        self.player.update()

    def update(self):
        """
        Updates the current world state by updating all objects in lanes.
        """
        # check if player on lilypad
        for lane in self.directed_lanes:
            lane.update()

    def check_vehicle_collision(self):
        """
        :return: True, if the player collides with a vehicle. False, otherwise
        """
        for street_lane in self.street_lanes:
            if isinstance(street_lane, DirectedLane) and collision_handler.check_collision_group(self.player,
                                                                                                 street_lane.non_player_sprites):
                return True

        return False

    def check_player_on_lilypad(self):
        """
        :return: A list with all sprites on which the player currently stands. An empty list, if they stand on no sprite.
        """
        lilypads = []
        for water_lane in self.water_lanes:
            if isinstance(water_lane, WaterLane):
                lilypads += spritecollide(self.player, water_lane.non_player_sprites, False, collided=pygame.sprite.collide_rect)

        return lilypads

    def check_water_collision(self):
        """
        :return: True, if the player does not collide with a lilypad and thus collides with the water.
        """
        # check collision with water
        on_water = collision_handler.check_collision_group(self.player, self.water_lanes)

        # check collision with lilypads (=water_lane.sprites) or other ground)
        on_ground = False
        if collision_handler.check_collision_group(self.player, self.middle_lanes):
            on_ground = True
        elif collision_handler.check_collision_group(self.player, self.finish_lanes):
            on_ground = True
        else:
            for water_lane in self.water_lanes:
                if isinstance(water_lane, DirectedLane):
                    if collision_handler.check_collision_group(self.player, water_lane.non_player_sprites):
                        on_ground = True
                        break
        return not on_ground and on_water

    def draw_lanes(self, screen) -> None:
        for lane in self.lanes.sprites():
            if isinstance(lane, Lane):
                lane.draw_lane(screen)

    def _generate_lanes(self) -> None:
        row = config.N_LANES - 1

        # starting lane
        if config.N_FIELDS_PER_LANE % 2 == 0:
            starting_position_x = config.N_FIELDS_PER_LANE // 2 - 1
        else:
            starting_position_x = config.N_FIELDS_PER_LANE // 2
        starting_position_y = config.N_LANES - 1

        starting_lane = StartLane(row, starting_position=(starting_position_x, starting_position_y))
        print(f"starting position = {starting_lane.starting_position}")
        self.starting_lanes.add(starting_lane)
        self.lanes.add(starting_lane)
        row -= 1

        # street lanes
        for i in range(config.N_STREET_LANES):
            direction = LaneDirection.LEFT if i % 2 == 0 else LaneDirection.RIGHT
            street_lane = StreetLane(row, direction)
            self.street_lanes.add(street_lane)
            self.directed_lanes.add(street_lane)
            self.lanes.add(street_lane)
            row -= 1

        # interim lane
        middle_lane = Lane(row)
        self.middle_lanes.add(middle_lane)
        self.lanes.add(middle_lane)
        row -= 1

        # water lanes
        for i in range(config.N_WATER_LANES):
            direction = LaneDirection.LEFT if i % 2 == 0 else LaneDirection.RIGHT
            water_lane = WaterLane(row, direction)
            self.water_lanes.add(water_lane)
            self.directed_lanes.add(water_lane)
            self.lanes.add(water_lane)
            row -= 1

        # finish lane
        finish_lane = FinishLane(row, 8)
        self.finish_lanes.add(finish_lane)
        self.lanes.add(finish_lane)

        assert row == 0, f"Error in lane generation, row={row}"

    def __json__(self):
        """Saves the given world as json at the given path."""
        world_dict = {}

        height = len(self.lanes)
        width = len(self.lanes.sprites()[0].fields)
        street_lanes = len(self.street_lanes)
        middle_lanes = len(self.middle_lanes)
        water_lanes = len(self.water_lanes)

        starting_position = self.starting_lanes.sprites()[0].starting_position
        target_position = self.finish_lanes.sprites()[0].target_position

        # rewrite lanes into dictionary
        lanes_list = []
        for lane in self.lanes:
            lane_dict = {}
            lane_dict['row'] = lane.row
            lane_dict['type'] = lane.__class__.__name__

            if isinstance(lane, DirectedLane):
                lane_dict['direction'] = lane.direction.value

            # TODO spawn parameters

            lanes_list.append(lane_dict)

        # write parameters to dictionary
        world_dict["height"] = height
        world_dict["width"] = width
        world_dict["street_lanes"] = street_lanes
        world_dict["middle_lanes"] = middle_lanes
        world_dict["water_lanes"] = water_lanes
        world_dict["starting_position"] = starting_position
        world_dict["target_position"] = target_position
        world_dict["lanes"] = lanes_list

        return world_dict
