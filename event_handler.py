import sys

import pygame

blocked = False


class EventHandler:

    def __init__(self, game):
        """ Initializes the event handler. """
        self.game = game
        self.world = game.world

    def handle_input_event(self):
        """Requests all recent events and handles them by returning the action the player has taken.
        :returns the delta directions in which the player is moving as (x, y)-tuple. E.g. moving to north is (0, -1),
        standing still is (0, 0) and moving to the bottom left is (-1, 1).
        """
        # check newly pressed keys (only once and latest event first)
        input_events = pygame.event.get([pygame.KEYUP, pygame.KEYDOWN])
        for input_event in reversed(input_events):
            if self.world.player.delta_y + self.world.player.delta_x != 0:
                break

            if input_event.type == pygame.KEYDOWN:
                # down
                if input_event.key == pygame.K_DOWN:
                    self.world.player.delta_y = 1

                # left
                if input_event.key == pygame.K_LEFT:
                    self.world.player.delta_x = -1

                # right
                if input_event.key == pygame.K_RIGHT:
                    self.world.player.delta_x = 1

                # up
                if input_event.key == pygame.K_UP:
                    self.world.player.delta_y = -1

        # process other non-movement-related events
        for input_event in input_events:
            if input_event.type == pygame.KEYUP:
                if input_event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if input_event.key == pygame.K_p:
                    self.game.pause = not self.game.pause
                if input_event.key == pygame.K_F1:
                    self.game.display_debug_information_objects = False
                    self.game.display_debug_information_player = not self.game.display_debug_information_player
                    self.game.display_debug_information_lanes = False
                if input_event.key == pygame.K_F2:
                    self.game.display_debug_information_player = False
                    self.game.display_debug_information_objects = not self.game.display_debug_information_objects
                    self.game.display_debug_information_lanes = False
                if input_event.key == pygame.K_F3:
                    self.game.display_debug_information_player = False
                    self.game.display_debug_information_objects = False
                    self.game.display_debug_information_lanes = not self.game.display_debug_information_lanes

        # check if key is still pressed
        pressed_keys = pygame.key.get_pressed()
        self.world.player.delta_x = pressed_keys[pygame.K_RIGHT] - pressed_keys[pygame.K_LEFT] if self.world.player.delta_y + self.world.player.delta_x == 0 else self.world.player.delta_x
        self.world.player.delta_y = pressed_keys[pygame.K_DOWN] - pressed_keys[pygame.K_UP] if self.world.player.delta_y + self.world.player.delta_x == 0 else self.world.player.delta_y
