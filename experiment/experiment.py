import pygame

import colors

pygame.init()

import config

from game import Game
from world_generation.generation_config import GameDifficulty

# TODO move to experiment config
N_WORLDS_PER_DIFFICULTY = 20


class Experiment:

    def __init__(self):
        self.current_game = None

        # create a window
        self.screen = pygame.display.set_mode((config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX), pygame.FULLSCREEN)
        self.screen.fill(colors.WHITE)

    def welcome_screen(self):
        """
        Shows welcome screen.
        """
        self.show_message("Welcome to the experiment! You will now play multiple levels of frogger with different difficulty.")
        self.show_message("We will start with the rules of the game and some example levels.", y_offset=100)
        self.show_message("Press any key to continue.", y_offset=200)
        wait_keys()

    def run(self):
        """ Runs the experiment. """

        self.welcome_screen()

        # run each level
        for difficulty in GameDifficulty:
            for i in range(N_WORLDS_PER_DIFFICULTY):
                # create game
                world_name = "{}/world_{}".format(difficulty.value, i)
                self.current_game = Game(difficulty, world_name, screen=self.screen)

                # run game
                pygame.event.clear()
                self.current_game.run()

                # log data
                # TODO

                # show score
                # TODO

                # show message between levels
                self.show_message_between_levels()

    def show_message_between_levels(self):
        """
        Shows message between levels.
        """
        self.show_message("You have finished the current level."),
        self.display_score(self.current_game.calculate_score())
        self.show_message("Press any key to continue to the next level.", y_offset=600)

    def display_score(self, score: dict):
        """ Displays the score. Requires the score as dict to provide a detailed explanation how the score is calculated"""
        # TODO
        self.show_message("SCORE", y_offset=200)

    def show_message(self, msg, y_offset=0):
        """
        Shows message on psychopy window

        :param y_offset: offset from top of text box
        :param wrap_width: width of the textbox
        :param msg: Message to show
        """
        drawText(self.screen, msg, colors.BLACK,
                 pygame.rect.Rect(config.DISPLAY_WIDTH_PX / 4, 200 + y_offset, config.DISPLAY_WIDTH_PX / 2, config.DISPLAY_HEIGHT_PX / 2),
                 font=pygame.font.SysFont(pygame.font.get_default_font(), 30))

        # self.screen.blit(text_surface, (200, 200))
        pygame.display.flip()


    def log_data(self):
        """ Logs the data. """
        # TODO
        pass


def wait_keys():
    """ Waits for a key to be pressed. """
    waiting = True
    while waiting:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            waiting = False


# draw some text into an area of a surface
# automatically wraps words
# returns any text that didn't get blitted
def drawText(surface, text, color, rect, font, aa=True, bkg=None):
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

        surface.blit(image, (rect.centerx - font.size(text)[0] / 2, y))
        y += fontHeight + lineSpacing

        # remove the text we just blitted
        text = text[i:]

    return text


exp = Experiment()
exp.run()
