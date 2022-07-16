import pygame
from pygaze._screen.pygamescreen import PyGameScreen
from pygaze import settings


class EyeTrackerScreen(PyGameScreen):

    def __init__(self, dispsize=settings.DISPSIZE, fgc=settings.FGC,
                 bgc=settings.BGC, mousevisible=settings.MOUSEVISIBLE, screen=None,
                 **args):
        super().__init__(dispsize=dispsize, fgc=fgc, bgc=bgc, mousevisible=mousevisible, screen=screen, **args)
        self.create()

    def fill(self, color):
        self.set_background_colour(color)
        self.create()
        self.clear()
        self.screen.fill(color)

    def blit(self, *args):
        self.screen.blit(*args)

    def draw_arc(self, color, rect, start_angle, end_angle, width=1):
        pygame.draw.arc(self.screen, color, rect, start_angle, end_angle, width)

