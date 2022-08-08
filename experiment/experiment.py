import csv
import os
import random
import sys

import numpy as np
import pygame

import colors
from experiment.experiment_config import *

from text_utils import drawText

import config

from game import Game
from world_generation.generation_config import GameDifficulty
from eye_tracker import MyEyeTracker


class Experiment:

    def __init__(self, eye_tracker=False):
        self.current_game = None

        # create pygaze Display object

        if eye_tracker:

            self.eye_tracker = MyEyeTracker()
            if self.eye_tracker.connected():
                print("Connection to eye tracker established successfully")

            self.eye_tracker.calibrate()

        else:
            self.eye_tracker = None

        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen.fill(colors.WHITE)
        pygame.init()

        self.subject_id = None
        self.subject_score = 0

        self.level_num = 1

    def _welcome_screen_template(self):
        """ Shows the welcome screen template. """
        self.screen.fill(colors.WHITE)
        self.show_message("Welcome to the experiment! You will now play multiple levels of frogger with different difficulty.")
        self.show_message("Please enter your subject ID first. It consists of the following parts:", y_offset=200, alignment="left")
        self.show_message("1) The first two letters of the first name of your first parent", y_offset=250, alignment="left")
        self.show_message("2) The number of your birth month (with leading zeroes)", y_offset=300, alignment="left")
        self.show_message("3) The first two letters of the first name of your second parent", y_offset=350, alignment="left")
        self.show_message("For example, the son of Claudia and Ralf, born on the 25.03.1998, will yield the code \'CL03RA\'.", y_offset=400,
                          alignment="left")
        self.show_message("Press enter to continue.", y_offset=600)

    def show_screen(self):
        pygame.display.update()

    def welcome_screen(self):
        """
        Shows the welcome screen and receives the subject id.
        """
        clock = pygame.time.Clock()

        text = ""
        input_active = True
        run = True
        while run:
            self._welcome_screen_template()
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    input_active = True
                    text = ""
                elif event.type == pygame.KEYDOWN and input_active:
                    if event.key == pygame.K_RETURN and len(text) == 6:
                        run = False
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif event.key == pygame.K_SPACE:
                        pass  # skip spaces since they are not allowed in the id
                    else:
                        text += event.unicode

            text = text.upper()

            self.show_message(text, y_offset=450)
            self.show_screen()
            self.subject_id = check_if_subject_id_exists(text)

    def strategy_screen(self):

        self.show_screen()
        clock = pygame.time.Clock()

        text_rows = [""]
        text_row_ptr = 0
        input_active = True
        run = True
        while run:
            self.screen.fill(colors.WHITE)
            self.show_message(
                "Did you pursue a certain strategy playing? Did you change this strategy? Please write some sentences on that.",
                alignment='left')
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    input_active = True
                elif event.type == pygame.KEYDOWN and input_active:
                    if event.key == pygame.K_RETURN:
                        text_row_ptr += 1
                        if text_row_ptr >= len(text_rows):
                            text_rows.append("")
                    elif event.key == pygame.K_DOWN:
                        text_row_ptr += 1
                    elif event.key == pygame.K_TAB:
                        input_active = False
                        run = False
                    elif event.key == pygame.K_UP:
                        text_row_ptr -= 1
                    elif event.key == pygame.K_BACKSPACE:
                        text_rows[text_row_ptr] = text_rows[text_row_ptr][:-1]
                    else:
                        text_rows[text_row_ptr] += event.unicode

            y_offset_base = 100
            for i, row_text in enumerate(text_rows):
                self.show_message(row_text, x=80, y_offset=y_offset_base + i * 30, width=0.9 * config.DISPLAY_WIDTH_PX, alignment='left')
            self.show_screen()

            self.show_message("Please press TAB to continue (this may take a moment)")
        wait_keys()

    def rules_screen(self):
        """ Shows the rules of the game. """
        self.screen.fill(colors.WHITE)
        example_level_img = pygame.image.load(r'./images/example_level.png')
        example_level_img = pygame.transform.scale(example_level_img, (config.DISPLAY_WIDTH_PX * 2 / 3, config.DISPLAY_HEIGHT_PX * 2 / 3))
        self.screen.blit(example_level_img, (config.DISPLAY_WIDTH_PX / 6, 50))

        self.show_message("You start as the frog at the bottom of the screen and your goal is to get to the star at the top.",
                          y_offset=600)
        self.show_message("You can move in any direction with the arrow keys.",
                          y_offset=640)
        self.show_message(
            "You need to avoid cars and water to get to the star, while staying inside the level borders.",
            y_offset=680)
        self.show_message("Also, pay attention to the timer indicated by the circle on the frog. If it runs out, you lose.", y_offset=720)
        self.show_message("You get points for remaining time, finishing the level and moving closer to the star.",
                          y_offset=760)
        self.show_message("We will start with some example levels to get you started. Press any key to continue.",
                          y_offset=800)

        self.show_screen()
        wait_keys()

    def pre_start_screen(self):
        self.screen.fill(colors.WHITE)

        self.show_message("Press SPACE to start!",
                          y_offset=300, font_size=100)

        self.show_screen()

        # wait for space bar
        wait_keys([pygame.K_SPACE])

        # show countdown
        self._show_countdown()

    def _show_countdown(self):
        """ Shows the countdown. """
        for i in reversed(range(1, 4)):
            self.screen.fill(colors.WHITE)
            self.show_message(str(i),
                              y_offset=300, font_size=100)
            self.show_screen()
            pygame.time.wait(1000)

    def training_explanation_screen(self):
        """ Explains that we will start with some training examples"""
        self.screen.fill(colors.WHITE)
        self.show_message("We will start with some training examples. Please press any key to continue.",
                          y_offset=300)

        self.show_screen()
        wait_keys()

    def experiment_start_screen(self):
        """ Explains that the real experiment will start now."""
        self.screen.fill(colors.WHITE)
        self.show_message("We will start with the real experiment now. Do not hesitate do ask any questions.",
                          y_offset=300)

        self.show_message("Please press any button to continue, when you are ready.",
                          y_offset=400)

        self.show_screen()
        wait_keys()

    def run_trial(self, difficulty, world_name, training=False):
        """Runs a trial / game for given difficulty and world."""
        self.current_game = Game(difficulty, world_name=world_name, eye_tracker=self.eye_tracker, screen=self.screen,
                                 subject_id=self.subject_id)

        # Show pre-start screen
        self.pre_start_screen()

        # run game
        self.current_game.run()

        # save logged data after level
        self.current_game.save_logging_data(training=training)

    def run(self):
        """ Runs the experiment. """

        # -------- INTRO ----------
        self.welcome_screen()
        self.rules_screen()

        # -------- TRAINING ----------
        self.training_explanation_screen()
        easy_training_games = [(GameDifficulty.EASY, 'world_{}'.format(i)) for i in range(N_EASY_TRAINING_GAMES)]
        normal_training_games = [(GameDifficulty.NORMAL, 'world_{}'.format(i)) for i in range(N_NORMAL_TRAINING_GAMES)]
        self.level_num = -(N_EASY_TRAINING_GAMES + N_NORMAL_TRAINING_GAMES)
        for difficulty, world_name in easy_training_games + normal_training_games:
            self.run_trial(difficulty, world_name, training=True)

            self.show_screen_between_levels_training()
            self.loading_screen()
            self.level_num += 1

        self.loading_screen()

        # -------- EXPERIMENT ----------
        self.experiment_start_screen()
        self.show_screen()

        # get all possible levels and shuffle them
        self.level_num = 1
        possible_games = [(difficulty, 'world_{}'.format(i)) for difficulty in GameDifficulty for i in range(N_WORLDS_PER_DIFFICULTY)]
        random.shuffle(possible_games)

        # run each level
        for difficulty, world_name in possible_games:
            self.run_trial(difficulty, world_name)

            # show screen between levels with score and further instructions
            self.show_screen_between_levels()
            self.loading_screen()
            self.level_num += 1

        # end screen
        self.end_screen()

    def loading_screen(self):
        self.screen.fill(colors.WHITE)
        self.show_message('LOADING...',
                          y_offset=300, font_size=100)
        self.show_screen()

    def end_screen(self):
        self.screen.fill(colors.WHITE)
        self.show_message('That\'s it, thank you very much for participating!',
                          y_offset=300, font_size=60)
        self.show_message('Please press any key to exit now.', y_offset=500)
        self.show_screen()
        wait_keys()

    def show_screen_between_levels(self):
        """
        Shows message between levels.
        """
        self.screen.fill(colors.WHITE)

        # show score of current level
        self.display_and_update_score(self.current_game.calc_score())
        self.show_message("Press any key to continue to the next level.", y_offset=600)

        self.show_screen()
        wait_keys()

        # show highscore screen and save current score after the first and each five levels
        if self.level_num == 1 or self.level_num % 5 == 0:
            # save own score
            self.save_score()

            # show highscore after that number of levels (maybe not after each level but only after each 10 or so)
            self.display_highscores()
            self.show_message("Press any key to continue to the next level.", y_offset=600)

            self.show_screen()
            wait_keys()

    def show_screen_between_levels_training(self):
        self.screen.fill(colors.WHITE)
        self.show_message("Press any key to continue to the next level.", y_offset=300)
        self.show_screen()
        wait_keys()

    def display_highscores(self):
        """
        Displays highscores.
        """
        self.screen.fill(colors.WHITE)

        # load score data
        scores = self.load_scores()

        self.show_message("HIGHSCORES")

        y_offset = 100
        for name, score in scores:
            self._draw_score_row(name, score, y_offset=y_offset)
            y_offset += 50

    def load_scores(self):
        """ Loads the score from the database. """

        # load from csv
        with open(config.SCORE_DIR + "after_{}_levels.csv".format(self.level_num), 'r') as f:
            reader = csv.reader(f, delimiter=';')
            scores = list(reader)

        scores.sort(key=lambda x: int(x[1]), reverse=True)  # sort in place by points
        scores = np.array(scores[:N_SCORES_DISPLAYED])

        return scores

    def save_score(self):
        """ Saves the score to the database. """

        # open the file in the write mode or create it before if it doesn't exist
        with open(config.SCORE_DIR + "after_{}_levels.csv".format(self.level_num), 'a', newline='') as f:
            # create the csv writer
            writer = csv.writer(f, delimiter=';')

            # write a row to the csv file
            writer.writerow([self.subject_id, self.subject_score])

    def display_and_update_score(self, score: dict):
        """ Displays the score. Requires the score as dict to provide a detailed explanation how the score is calculated"""

        level_score = (np.sum(list(score.values())) - score['difficulty_multiplier'])
        total_level_score = int(level_score * score['difficulty_multiplier'] if score['death_penalty'] == 0 else level_score)
        self.subject_score += total_level_score

        self.show_message("SCORE")
        y_offset = 50

        # Win Bonus
        self._draw_score_row('Win Bonus', str(score['win_bonus']), y_offset)
        y_offset += 50

        # Death Penalty
        self._draw_score_row('Death Penalty', str(score['death_penalty']), y_offset)
        y_offset += 50

        # Remaining Time
        self._draw_score_row('Remaining Time', str(score['remaining_time']), y_offset)
        y_offset += 50

        # Visited Lanes
        self._draw_score_row('Visited Lanes', str(score['visited_lanes']), y_offset)
        y_offset += 25

        # draw line
        pygame.draw.line(self.screen, colors.BLACK, (config.DISPLAY_WIDTH_PX / 5, 200 + y_offset),
                         (4 * config.DISPLAY_WIDTH_PX / 5, 200 + y_offset), 2)
        y_offset += 15

        self._draw_score_row('Level Score', str(level_score), y_offset)
        y_offset += 50

        self._draw_score_row('Difficulty Multiplier', 'X' + str(score['difficulty_multiplier']), y_offset)
        y_offset += 50

        # draw line
        pygame.draw.line(self.screen, colors.BLACK, (config.DISPLAY_WIDTH_PX / 5, 200 + y_offset),
                         (4 * config.DISPLAY_WIDTH_PX / 5, 200 + y_offset), 2)

        y_offset += 15
        self._draw_score_row('Total Level Score', str(total_level_score), y_offset)

        y_offset += 100

        self._draw_score_row('Your Total Score is now:', str(self.subject_score), y_offset)

    def _draw_score_row(self, row_name, points, y_offset=0, font_size=30):
        """ Draws a row with the given name (aligned left) and points (aligned right). """
        drawText(self.screen, row_name, colors.BLACK,
                 pygame.rect.Rect(config.DISPLAY_WIDTH_PX / 5, 200 + y_offset, config.DISPLAY_WIDTH_PX / 5, font_size),
                 font_size=font_size, alignment='left')
        drawText(self.screen, points, colors.BLACK,
                 pygame.rect.Rect(3 * config.DISPLAY_WIDTH_PX / 5, 200 + y_offset, config.DISPLAY_WIDTH_PX / 5, font_size),
                 font_size=font_size, alignment='right')

    def show_message(self, msg, x=config.DISPLAY_WIDTH_PX / 4, width=config.DISPLAY_WIDTH_PX / 2, y_offset=0, font_size=30,
                     alignment='center'):
        """
        Shows message on psychopy window

        :param x: position of the textbox
        :param width: width of the textbox
        :param alignment: alignment of text
        :param font_size: size of the font to show the message
        :param y_offset: offset from top of text box
        :param msg: Message to show
        """
        drawText(self.screen, msg, colors.BLACK,
                 pygame.rect.Rect(x, 200 + y_offset, width, config.DISPLAY_HEIGHT_PX / 2),
                 font_size=font_size, alignment=alignment)


def check_if_subject_id_exists(subject_id):
    """ Checks if the subject id already exists and returns a modified subject id if it does. """
    cnt = 0
    for file in os.listdir(config.LEVEL_DATA_DIR):
        # check if file is directory
        if os.path.isdir(os.path.join(config.LEVEL_DATA_DIR, file)):
            if file.startswith(subject_id):
                cnt += 1
    return subject_id + '_' + str(cnt) if cnt > 0 else subject_id


def wait_keys(keys=None):
    """ Waits for a key to be pressed. """
    waiting = True
    while waiting:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:

            # close program if escape is pressed
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if keys is None or event.key in keys:
                waiting = False
