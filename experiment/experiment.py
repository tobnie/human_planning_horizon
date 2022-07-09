import csv
import sys

import numpy as np
import pygame
import colors

pygame.init()

import config

from game import Game
from world_generation.generation_config import GameDifficulty

# TODO move to experiment config
N_WORLDS_PER_DIFFICULTY = 20
N_SCORES_DISPLAYED = 10


class Experiment:

    def __init__(self):
        self.current_game = None

        # TODO
        # # create pygaze Display object
        # disp = libscreen.Display(disptype='pygame')
        # pygaze_screen = libscreen.Screen(disptype='pygame', screen=disp)

        self.subject_id = "DUMMY"  # TODO
        self.subject_score = 0

        self.level_num = 1

        # create a window
        self.screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)
        self.screen.fill(colors.WHITE)

    def welcome_screen(self):
        """
        Shows welcome screen.
        """
        self.screen.fill(colors.WHITE)
        self.show_message("Welcome to the experiment! You will now play multiple levels of frogger with different difficulty.")
        self.show_message("We will start with the rules of the game and some example levels.", y_offset=100)
        self.show_message("Press any key to continue.", y_offset=200)
        pygame.display.flip()
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

        pygame.display.flip()
        wait_keys()

    def pre_start_screen(self):
        self.screen.fill(colors.WHITE)

        self.show_message("Press SPACE to start!",
                          y_offset=300, font_size=100)

        pygame.display.flip()

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
            pygame.display.flip()
            pygame.time.wait(1000)

    def run(self):
        """ Runs the experiment. """

        self.welcome_screen()
        self.rules_screen()

        # run each level
        for difficulty in GameDifficulty:
            for i in range(N_WORLDS_PER_DIFFICULTY):
                # create game
                world_name = "world_{}".format(i)
                self.current_game = Game(difficulty, world_name, screen=self.screen, subject_id=self.subject_id)

                # Show pre-start screen
                self.pre_start_screen()

                # run game
                self.current_game.run()

                # save logged data after level
                # TODO maybe loading time or similar if it takes too long?
                self.current_game.save_logging_data()

                # show screen between levels with score and further instructions
                self.show_screen_between_levels()
                self.level_num += 1

    def show_screen_between_levels(self):
        """
        Shows message between levels.
        """
        self.screen.fill(colors.WHITE)

        # show score of current level
        self.display_and_update_score(self.current_game.calc_score())
        self.show_message("Press any key to continue to the next level.", y_offset=600)

        pygame.display.flip()
        wait_keys()

        # save own score
        self.save_score()

        # show highscore after that number of levels (maybe not after each level but only after each 10 or so)
        self.display_highscores()
        self.show_message("Press any key to continue to the next level.", y_offset=600)

        pygame.display.flip()
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

        print(scores)
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
        total_level_score = level_score * score['difficulty_multiplier']
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
                 font=pygame.font.SysFont(pygame.font.get_default_font(), font_size), alignment='left')
        drawText(self.screen, points, colors.BLACK,
                 pygame.rect.Rect(3 * config.DISPLAY_WIDTH_PX / 5, 200 + y_offset, config.DISPLAY_WIDTH_PX / 5, font_size),
                 font=pygame.font.SysFont(pygame.font.get_default_font(), font_size), alignment='right')

    def show_message(self, msg, y_offset=0, font_size=30):
        """
        Shows message on psychopy window

        :param font_size: size of the font to show the message
        :param y_offset: offset from top of text box
        :param wrap_width: width of the textbox
        :param msg: Message to show
        """
        drawText(self.screen, msg, colors.BLACK,
                 pygame.rect.Rect(config.DISPLAY_WIDTH_PX / 4, 200 + y_offset, config.DISPLAY_WIDTH_PX / 2, config.DISPLAY_HEIGHT_PX / 2),
                 font=pygame.font.SysFont(pygame.font.get_default_font(), font_size))


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


# draw some text into an area of a surface
# automatically wraps words
# returns any text that didn't get blitted
def drawText(surface, text, color, rect, font, aa=True, bkg=None, alignment='center'):
    rect = pygame.Rect(rect)
    y = rect.top
    lineSpacing = -2

    # get the height of the font
    fontHeight = font.size("Tg")[1]

    while text:
        i = 1

        # determine if the row of text will be outside our area
        if y + fontHeight > rect.bottom:
            break

        # determine maximum width of line
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        # if we've wrapped the text, then adjust the wrap to the last word
        if i < len(text):
            i = text.rfind(" ", 0, i) + 1

        # render the line and blit it to the surface
        if bkg:
            image = font.render(text[:i], 1, color, bkg)
            image.set_colorkey(bkg)
        else:
            image = font.render(text[:i], aa, color)

        if alignment == 'center':
            blit_x = rect.centerx - font.size(text)[0] / 2
        elif alignment == 'left':
            blit_x = rect.x
        elif alignment == 'right':
            blit_x = rect.right - font.size(text)[0]
        else:
            raise Exception("Unknown alignment: {}".format(alignment))

        surface.blit(image, (blit_x, y))
        y += fontHeight + lineSpacing

        # remove the text we just blitted
        text = text[i:]

    return text


exp = Experiment()
exp.run()
