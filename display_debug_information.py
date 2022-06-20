import string

import pygame

import colors
import config
from world.lane import DirectedLane
from world.world import WorldStatus


class TextDisplayer:

    def __init__(self, game):
        self.game = game
        self.screen = game.screen
        self.world = game.world

        self.font_color = colors.WHITE
        self.font_size = config.font_size
        self.font = pygame.freetype.SysFont(name="freesansbold", size=self.font_size)

    def display_debug_information(self):
        if self.game.display_debug_information_player:
            self.render_debug_information(self.debug_information_player())
        if self.game.display_debug_information_objects:
            self.render_debug_information(self.debug_information_objects())
        if self.game.display_debug_information_lanes:
            self.render_debug_information(self.debug_information_lanes())

    def render_debug_information(self, debug_info):
        for i, debug_line in enumerate(debug_info):
            text_surface, rect = self.font.render(debug_line, self.font_color)
            self.screen.blit(text_surface, (0, i * self.font_size))

    def debug_information_player(self) -> string:
        debug_information = [f"Player Position = ({self.world.player.x}, {self.world.player.y})",
                             f"Player Delta = ({self.world.player.delta_x}, {self.world.player.delta_y})",
                             f"Vehicle Collision = {self.game.world.player.check_vehicle_collision()}",
                             f"Water Collision = {self.game.world.player.check_water_collision()}",
                             "DEAD" if self.world.player.is_dead else "ALIVE"]
        if self.game.world_status == WorldStatus.WON:
            debug_information.append("GAME WON!")

        return debug_information

    def debug_information_objects(self) -> string:
        debug_information = []
        for i, lane in enumerate(self.game.world.lanes):
            debug_information.append("")
            if isinstance(lane, DirectedLane):
                sprite_info = [f"(ID: {s.id}, x:{s.x}, y:{s.y}, delta:{s.delta_x})" for s in lane.non_player_sprites]
                debug_information.append(f"{lane.__class__} ({lane.row}) (len={len(lane)})")
                debug_information.append(f"Objects: {', '.join(sprite_info)}")
            else:
                debug_information.append(f"{lane.__class__} ({lane.row})")
                debug_information.append("Empty")
        return debug_information

    def debug_information_lanes(self) -> string:
        debug_information = []
        for i, lane in enumerate(self.game.world.lanes):
            debug_information.append("")
            if isinstance(lane, DirectedLane):
                debug_information.append(
                    f"{lane.__class__} row={lane.row} len={len(lane)} v={lane.velocity} obst_size={lane.obstacle_size}"
                    f"d={lane.distance_between_obstacles} gap_skip={lane.obstacles_without_gap}")
            else:
                debug_information.append(f"{lane.__class__} row={lane.row} len={len(lane)}")

        return debug_information
