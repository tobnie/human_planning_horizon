import sys

import pygame

blocked = False


def handle_events(game):
    """Requests all recent events and handles them by returning the action the player has taken.
    :returns the delta directions in which the player is moving as (x, y)-tuple. E.g. moving to north is (0, -1),
    standing still is (0, 0) and moving to the bottom left is (-1, 1).
    """

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                game.world.player.delta_y = -1
            elif event.key == pygame.K_RIGHT:
                game.world.player.delta_x = 1
            elif event.key == pygame.K_DOWN:
                game.world.player.delta_y = 1
            elif event.key == pygame.K_LEFT:
                game.world.player.delta_x = -1

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                game.world.player.delta_y = 0
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                game.world.player.delta_x = 0
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_p:
                game.pause = not game.pause
            if event.key == pygame.K_F1:
                game.display_debug_information_objects = False
                game.display_debug_information_player = not game.display_debug_information_player
            if event.key == pygame.K_F2:
                game.display_debug_information_player = False
                game.display_debug_information_objects = not game.display_debug_information_objects
