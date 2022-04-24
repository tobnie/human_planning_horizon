import random

import pygame

import colors
import config
from world.lane import StartLane, StreetLane, WaterLane, FinishLane, Lane, LaneDirection, DirectedLane
from world.player import Player


class World:

    def __init__(self, width, height):
        # game boundaries
        self.width = width
        self.height = height

        # player
        self.player = Player(self)  # spawn player
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)

        # lanes
        self.lanes = self._generate_lanes()

    def draw(self, screen) -> None:
        screen.fill(colors.BLACK)

        # draw lanes
        self.draw_lanes(screen)

        # draw player
        self.player_list.draw(screen)
        pygame.display.flip()

    def update(self):
        self.player.update()
        for lane in self.get_directed_lanes():
            lane.update()

    def draw_lanes(self, screen) -> None:
        for lane in self.lanes:
            lane.draw_lane(screen)

    def get_start_lane(self) -> StartLane:
        return self.lanes[-1]

    def get_finish_lane(self) -> FinishLane:
        return self.lanes[0]

    def get_street_lanes(self) -> [StreetLane]:
        return [lane for lane in self.lanes if isinstance(lane, StreetLane)]

    def get_directed_lanes(self) -> [DirectedLane]:
        return [lane for lane in self.lanes if isinstance(lane, DirectedLane)]

    def get_water_lanes(self) -> [WaterLane]:
        return [lane for lane in self.lanes if isinstance(lane, WaterLane)]

    @staticmethod
    def _generate_lanes() -> [Lane]:
        row = config.N_LANES - 1
        lanes = []

        # starting lane
        starting_lane = StartLane(row, 4)  #
        lanes.append(starting_lane)
        row -= 1

        # street lanes
        for i in range(config.N_STREET_LANES):
            direction = random.choice([LaneDirection.LEFT, LaneDirection.RIGHT])
            street_lane = StreetLane(row, direction)
            lanes.append(street_lane)
            row -= 1

        # interim lane
        middle_lane = Lane(row)
        lanes.append(middle_lane)
        row -= 1

        # water lanes
        for i in range(config.N_WATER_LANES):
            direction = random.choice([LaneDirection.LEFT, LaneDirection.RIGHT])
            water_lane = WaterLane(row, direction)
            lanes.append(water_lane)
            row -= 1

        # finish lane
        finish_lane = FinishLane(row, 8)
        lanes.append(finish_lane)

        assert row == 0, f"Error in lane generation, row={row}"

        return lanes
