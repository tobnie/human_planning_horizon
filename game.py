import numpy as np
import pygame

from display_debug_information import TextDisplayer

import colors
import config
import event_handler
from world.world import World, WorldStatus

INPUT_EVENT = pygame.USEREVENT + 0
UPDATE_OBSTACLES_EVENT = pygame.USEREVENT + 1
UPDATE_PLAYER_EVENT = pygame.USEREVENT + 2
SPAWN_EVENT = pygame.USEREVENT + 3


class Game:

    def __init__(self, world_name: str = None, game_time=config.LEVEL_TIME):
        """
        Sets up the game by initializing PyGame.
        """
        # game clock
        self.clock = pygame.time.Clock()
        self.game_time = game_time

        # audio cue
        self.audio_cue_played = False

        self.display_debug_information_player = True
        self.display_debug_information_objects = False
        self.display_debug_information_lanes = False
        self.running = True
        self.pause = False
        self.fps = config.FPS

        self.world_status = WorldStatus.RUNNING
        self.world_name = world_name

        # collision counter
        self.vehicle_collision = False
        self.water_collision = False
        self.spawn_counter = 1

        # set screen information
        self.screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)

        if world_name:
            self.world = World(self, world_name=world_name)
        else:
            self.world = World(self, config.N_FIELDS_PER_LANE, config.N_LANES, )

        self.screen.fill(colors.BLACK)

        pygame.time.set_timer(UPDATE_OBSTACLES_EVENT, config.OBSTACLE_UPDATE_RATE)
        pygame.time.set_timer(UPDATE_PLAYER_EVENT, config.PLAYER_UPDATE_RATE)
        pygame.time.set_timer(SPAWN_EVENT, config.SPAWN_RATE)

        self.event_handler = event_handler.EventHandler(self)
        self.text_displayer = TextDisplayer(self)

    def run_pause(self):
        self.event_handler.handle_input_event()

    def run_normal(self):

        for e in pygame.event.get(exclude=[pygame.KEYUP, pygame.KEYDOWN]):
            if e.type == UPDATE_OBSTACLES_EVENT:
                self.world.update()
                if self.spawn_counter == config.SPAWN_RATE:
                    self.world.spawn()
                    self.spawn_counter = 1
                else:
                    self.spawn_counter += 1

            elif e.type == UPDATE_PLAYER_EVENT:
                self.event_handler.handle_input_event()

        self.world.update_player()
        self.render()
        self.game_time -= self.clock.tick_busy_loop(self.fps)

    def run(self):
        """
        Main Loop
        """
        while self.running:

            # run next game step
            if self.pause:
                self.run_pause()
            else:
                self.run_normal()

            self.world.check_game_state()
            if self.world_status == WorldStatus.WON:
                # game won
                self.start_world(self.world_name)
                # self.running = False
            if self.world_status == WorldStatus.LOST:
                # game lost
                self.start_world(self.world_name)
                # self.running = False

    def start_world(self, world_name):
        self.world = World(self, world_name=world_name)
        self.text_displayer = TextDisplayer(self)

    def check_timeout(self):
        """
        Checks if the game is over.
        """
        if self.game_time <= 0:
            self.world_status = WorldStatus.TIMED_OUT
            # self.running = False

    def draw_timer(self):
        """
        Draws the remaining time as a decreasing circle.
        """
        ratio_time_left = self.game_time / config.LEVEL_TIME
        pygame.draw.arc(self.screen, colors.RED,
                        (self.world.player.rect.x, self.world.player.rect.y, config.FIELD_WIDTH, config.FIELD_HEIGHT), 0,
                        np.deg2rad(360 * ratio_time_left), 15)

        if self.game_time < config.LEVEL_TIME_AUDIO_CUE and not self.audio_cue_played:
            self.audio_cue_played = True
            pygame.mixer.init()
            pygame.mixer.music.load(config.FROG_SOUND_FILEPATH)
            pygame.mixer.music.play()

    def render(self):
        """Renders the whole game."""
        self.world.draw(self.screen)

        self.draw_timer()
        self.text_displayer.display_debug_information()

        pygame.display.flip()
