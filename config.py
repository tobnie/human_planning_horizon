import pygame

# screen information
DISPLAY_WIDTH_PX = pygame.display.Info().current_w
DISPLAY_HEIGHT_PX = pygame.display.Info().current_h

# UI
font_size = 24

# world configuration
N_LANES = 15
N_STREET_LANES = 6
N_WATER_LANES = 6

N_FIELDS_PER_LANE = 20  # in original game 14 I guess
FIELD_WIDTH = DISPLAY_WIDTH_PX / N_FIELDS_PER_LANE
FIELD_HEIGHT = DISPLAY_HEIGHT_PX / N_LANES

# obstacles
OBSTACLE_VELOCITY = 1
OBSTACLE_SPAWN_RATE = 54
PLAYER_UPDATE_RATE = 6

# game dynamics
# PLAYER_VELOCITY = 1.0   # (px) e.g. for moving
FPS = 60  # frame rate

# directory
SPRITES_DIR = "./sprites/"
LEVELS_DIR = "./levels/"

# sprite files
LILYPAD_FILE = 'lilypad.jpg'
LOG_FILE = 'log.png'
