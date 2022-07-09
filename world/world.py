import json
from enum import Enum

import pygame

import colors
import config
from world.lane import StartLane, StreetLane, WaterLane, FinishLane, Lane, LaneDirection, DirectedLane
from world.player import Player
from world.world_state import WorldState


class WorldStatus(Enum):
    RUNNING = "Game Running."
    WON = "Game Won!"
    LOST = "Game Over"
    TIMED_OUT = "Time's up!"


class World:

    def __init__(self, game, width: int = None, height: int = None, world_path: str = None) -> None:
        self.game = game

        # game boundaries
        self.width = width if width is not None else 1
        self.height = height if height is not None else 1

        # lanes
        self.lanes = pygame.sprite.Group()
        self.directed_lanes = pygame.sprite.Group()
        self.starting_lanes = pygame.sprite.Group()
        self.street_lanes = pygame.sprite.Group()
        self.middle_lanes = pygame.sprite.Group()
        self.water_lanes = pygame.sprite.Group()
        self.finish_lanes = pygame.sprite.Group()

        if world_path:
            self.load_lanes_from_json(world_path)
        else:
            # create lanes
            self._generate_random_lanes()

        # create player
        self.player = Player(self)  # spawn player
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)

        # start position
        starting_lane: StartLane = self.starting_lanes.sprites()[0]
        player_start_x = starting_lane.starting_position[0] * config.FIELD_WIDTH
        player_start_y = starting_lane.starting_position[1] * config.FIELD_HEIGHT
        self.player.set_position((player_start_x, player_start_y))

    def draw(self, screen) -> None:
        """ Draws the world on the screen. """
        screen.fill(colors.BLACK)

        # draw lanes
        self.draw_lanes(screen)

        # draw player
        self.player_list.draw(screen)

    def spawn(self):
        """ Starts the check if a new entity should spawn for each directed lane. """
        for lane in self.directed_lanes.sprites():
            if isinstance(lane, DirectedLane):
                lane.spawn_entity()

    def update(self):
        """
        Updates the obstacle rects of all lanes
        """
        self.player.update()
        for lane in self.directed_lanes.sprites():
            if isinstance(lane, DirectedLane):
                lane.update()

    def check_game_state(self):
        """ Checks if the game has been won or lost. """
        # check if player is dead and end game
        if self.player.is_dead:
            # show screen for restart
            return WorldStatus.LOST

        if self._check_game_won():
            return WorldStatus.WON

        return WorldStatus.RUNNING

    def _check_game_won(self):
        """
        Checks if the game has been won.
        """
        finish_lane: FinishLane = self.finish_lanes.sprites()[0]
        if self.player.rect.y == 0 and finish_lane.target_position * config.FIELD_WIDTH <= self.player.rect.x <= (
                finish_lane.target_position + 1) * config.FIELD_WIDTH:
            return True

    def draw_lanes(self, screen) -> None:
        """ Draws all lanes on the screen. """
        for lane in self.lanes.sprites():
            if isinstance(lane, Lane):
                lane.draw_lane(screen)

    def _generate_random_lanes(self) -> None:
        """ Randomly generates lanes. """
        row = config.N_LANES - 1

        # starting lane
        if config.N_FIELDS_PER_LANE % 2 == 0:
            starting_position_x = config.N_FIELDS_PER_LANE // 2 - 1
        else:
            starting_position_x = config.N_FIELDS_PER_LANE // 2
        starting_position_y = config.N_LANES - 1

        starting_lane = StartLane(self, row, starting_position=(starting_position_x, starting_position_y))
        print(f"starting position = {starting_lane.starting_position}")
        self.starting_lanes.add(starting_lane)
        self.lanes.add(starting_lane)
        row -= 1

        # street lanes
        for i in range(config.N_STREET_LANES):
            direction = LaneDirection.LEFT if i % 2 == 0 else LaneDirection.RIGHT
            street_lane = StreetLane(self, row, direction)
            self.street_lanes.add(street_lane)
            self.directed_lanes.add(street_lane)
            self.lanes.add(street_lane)
            row -= 1

        # interim lane
        middle_lane = Lane(self, row)
        self.middle_lanes.add(middle_lane)
        self.lanes.add(middle_lane)
        row -= 1

        # water lanes
        for i in range(config.N_WATER_LANES):
            direction = LaneDirection.LEFT if i % 2 == 0 else LaneDirection.RIGHT
            water_lane = WaterLane(self, row, direction)
            self.water_lanes.add(water_lane)
            self.directed_lanes.add(water_lane)
            self.lanes.add(water_lane)
            row -= 1

        # finish lane
        if config.N_FIELDS_PER_LANE % 2 == 0:
            target_position = config.N_FIELDS_PER_LANE // 2 - 1
        else:
            target_position = config.N_FIELDS_PER_LANE // 2
        finish_lane = FinishLane(self, row, target_position)
        self.finish_lanes.add(finish_lane)
        self.lanes.add(finish_lane)

        assert row == 0, f"Error in lane generation, row={row}"

    def load_lanes_from_json(self, world_path: str) -> None:
        """ Loads lanes from a json file with the specified world name. """
        # load json at given path
        with open(config.LEVELS_DIR + world_path + '.json', 'r', encoding='utf-8') as f:
            world_dict = json.load(f)

        # create world
        # game boundaries
        self.width = world_dict['width']
        self.height = world_dict['height']
        starting_position = world_dict['starting_position']

        # parse lanes
        for lane_info in world_dict['lanes']:
            # start lane
            if lane_info['type'] == 'StartLane':
                starting_lane = StartLane(self, self.height - 1, starting_position=(starting_position[0], starting_position[1]))
                self.starting_lanes.add(starting_lane)
                self.lanes.add(starting_lane)
            # street lane
            elif lane_info['type'] == 'StreetLane':
                street_lane = StreetLane(self, lane_info['row'], LaneDirection(lane_info['direction']),
                                         velocity=lane_info['lane_velocity'],
                                         distance_between_obstacles=lane_info['distance_between_obstacles'],
                                         obstacle_size=lane_info['obstacle_size'],
                                         obstacles_without_gap=lane_info['obstacles_without_gap'])
                self.street_lanes.add(street_lane)
                self.directed_lanes.add(street_lane)
                self.lanes.add(street_lane)
            # finish lane
            elif lane_info['type'] == 'FinishLane':
                finish_lane = FinishLane(self, lane_info['row'], world_dict["target_position"])
                self.finish_lanes.add(finish_lane)
                self.lanes.add(finish_lane)
            # water lane
            elif lane_info['type'] == 'WaterLane':
                water_lane = WaterLane(self, lane_info['row'], LaneDirection(lane_info['direction']),
                                       velocity=lane_info['lane_velocity'],
                                       distance_between_obstacles=lane_info['distance_between_obstacles'],
                                       obstacle_size=lane_info['obstacle_size'],
                                       obstacles_without_gap=lane_info['obstacles_without_gap'])
                self.water_lanes.add(water_lane)
                self.directed_lanes.add(water_lane)
                self.lanes.add(water_lane)
            # interim lane
            elif lane_info['type'] == 'Lane':
                middle_lane = Lane(self, lane_info['row'])
                self.middle_lanes.add(middle_lane)
                self.lanes.add(middle_lane)
            else:
                raise Exception("Unknown lane type: {}".format(lane_info['type']))

    def get_world_state(self):
        return WorldState(self)

    def __json__(self):
        """Returns a json (dict) representation of the world."""
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
            lane_dict = {'row': lane.row, 'type': lane.__class__.__name__}

            # spawn parameters
            if isinstance(lane, DirectedLane):
                lane_dict['direction'] = lane.direction.value
                lane_dict['velocity'] = lane.velocity
                lane_dict['obstacle_size'] = lane.obstacle_size
                lane_dict['distance_between_obstacles'] = lane.distance_between_obstacles
                lane_dict['obstacles_without_gap'] = lane.obstacles_without_gap

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
