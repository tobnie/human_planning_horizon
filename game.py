import pygame
import pygame.gfxdraw
from pygaze import libtime

from display_debug_information import TextDisplayer

import colors
import config
import event_handler
from eye_tracker import MyEyeTracker
from text_utils import drawText
from logging_game.logger import Logger
from world.world import World, WorldStatus
from world_generation.generation_config import GameDifficulty

INPUT_EVENT = pygame.USEREVENT + 0
UPDATE_PLAYER_EVENT = pygame.USEREVENT + 1


class Game:

    def __init__(self, difficulty: GameDifficulty, eye_tracker: MyEyeTracker = None, world_name: str = None, time_limit=config.LEVEL_TIME,
                 screen=None,
                 subject_id=""):
        """
        Sets up the game by initializing PyGame.
        """
        # game clock
        self.clock = None
        self.time_limit = time_limit
        self.game_time = 0
        self.eye_tracker = eye_tracker

        # game difficulty
        self.difficulty = difficulty

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

        # spawn counter
        self.spawn_counter = 1

        # set screen information
        self.screen = screen

        if world_name:
            self.world = World(self, world_path=difficulty.value + '/' + world_name)
        else:
            self.world = World(self, config.N_FIELDS_PER_LANE, config.N_LANES, )

        self.screen.fill(colors.BLACK)

        self.event_handler = event_handler.EventHandler(self)
        self.text_displayer = TextDisplayer(self)

        # logging_game
        self.logger = Logger(self, subject_id, world_name, difficulty, time_limit)

    def reset_clock(self):
        self.clock = pygame.time.Clock()
        self.game_time = 0

    def run_pause(self):
        """ Runs the game in pause mode. """
        self.event_handler.handle_input_event()

    def pre_run(self):
        """ Pre-runs the game such that obstacles are already on the lanes and not just starting to spawn for the first time."""

        for i in range(config.PRE_RUN_ITERATIONS):
            self.world.update()

    def run_normal(self):
        """ Runs the game normally. """
        if self.eye_tracker:
            sample = self.eye_tracker.get_sample()
            time = sample[0]
            self.logger.log_eyetracker_samples(time, sample[1:])
            self.eye_tracker.extract_events()  # TODO check this
        else:
            time = self.game_time

        for e in pygame.event.get(exclude=[pygame.KEYUP, pygame.KEYDOWN]):
            if e.type == UPDATE_PLAYER_EVENT:
                self.event_handler.handle_input_event()
                self.logger.log_action(time)  # actions are logged every time a update_player_event occurs

        # update and render world
        self.world.update()
        self.render()

        # update game_time
        dt = self.clock.tick_busy_loop(self.fps)
        self.game_time += dt

        # log world state
        self.logger.log_state(time)

        # check world status
        self.world.player.check_status()

    def run(self):
        """
        Main Loop
        """
        self.pre_run()
        pygame.event.clear()
        pygame.time.set_timer(UPDATE_PLAYER_EVENT, config.PLAYER_UPDATE_INTERVAL)
        self.render()

        # reset times
        self.reset_clock()  # game time
        if self.eye_tracker:
            libtime.expstart()  # libtime
            self.eye_tracker.start_recording()  # eye tracker clock
        while self.running:

            # run next game step
            if self.pause:
                self.run_pause()
            else:
                self.run_normal()

            self.world_status = self.world.check_game_state()
            if self.world_status != WorldStatus.RUNNING:
                # draw text whether game was won, lost or time is up
                drawText(self.screen, self.world_status.value, colors.DARK_GREEN if self.world_status == WorldStatus.WON else colors.RED,
                         pygame.rect.Rect(config.DISPLAY_WIDTH_PX / 4, 7 * config.FIELD_HEIGHT, config.DISPLAY_WIDTH_PX / 2,
                                          config.DISPLAY_HEIGHT_PX / 2), font_size=90)

                self.flip_display()

                # game won
                pygame.time.wait(config.DELAY_AFTER_LEVEL_FINISH)
                self.running = False

        # stop eye tracker recording and flush event buffer
        if self.eye_tracker:
            self.eye_tracker.stop_recording()
            self.logger.log_eyetracker_events(self.eye_tracker.eyetracker_events)
            print(self.eye_tracker.eyetracker_events)
        pygame.event.clear()

    def save_logging_data(self, training=False):
        """ Saves the data in the game logger. """
        self.logger.save_data(training=training)

    def check_timeout(self):
        """
        Checks if the game is over.
        """
        if self.game_time >= self.time_limit:
            self.world_status = WorldStatus.TIMED_OUT

    def flip_display(self):
        pygame.display.flip()

    def draw_timer(self):
        """
        Draws the remaining time as a decreasing circle.
        """
        time_left = self.time_limit - self.game_time
        ratio_time_left = time_left / config.LEVEL_TIME

        offset_y = 5 / 8 * config.PLAYER_HEIGHT
        height = 15

        pygame.draw.rect(self.screen, colors.RED, (
            self.world.player.rect.x, self.world.player.rect.y + offset_y,
            ratio_time_left * config.PLAYER_WIDTH,
            height))
        pygame.draw.rect(self.screen, colors.BLACK, (
            self.world.player.rect.x, self.world.player.rect.y + offset_y, config.PLAYER_WIDTH, height), 3)

    def render(self):
        """Renders the whole game."""
        self.world.draw(self.screen)

        self.draw_timer()
        self.text_displayer.display_debug_information()

        self.flip_display()

    def calc_score(self):
        """
        Calculates the score of the game.
        """

        score = {
        }

        # death penalty
        if self.world_status == WorldStatus.LOST:
            score['death_penalty'] = config.DEATH_PENALTY
        else:
            score['death_penalty'] = 0

        # points for remaining time
        if not self.world_status.TIMED_OUT:
            score['remaining_time'] = self.game_time
        else:
            score['remaining_time'] = 0

        # flat bonus for winning
        if self.world_status == WorldStatus.WON:
            score['win_bonus'] = config.WIN_BONUS
        else:
            score['win_bonus'] = 0

        # points for each visited lane
        if self.world_status == WorldStatus.LOST:
            score['visited_lanes'] = 0
        else:
            score['visited_lanes'] = 5 * self.world.player.highest_visited_lane

        # point multiplier for difficulty
        if self.world_status == WorldStatus.LOST:
            score['difficulty_multiplier'] = 0
        elif self.difficulty == GameDifficulty.EASY:
            score['difficulty_multiplier'] = config.EASY_MULTIPLIER
        elif self.difficulty == GameDifficulty.NORMAL:
            score['difficulty_multiplier'] = config.NORMAL_MULTIPLIER
        elif self.difficulty == GameDifficulty.HARD:
            score['difficulty_multiplier'] = config.HARD_MULTIPLIER

        return score
