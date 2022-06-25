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
SPAWN_RATE = 2
OBSTACLE_UPDATE_RATE = 1000
PLAYER_UPDATE_RATE = 100

# game dynamics
FPS = 60  # frame rate
LEVEL_TIME = 60_000  # in ms
LEVEL_TIME_AUDIO_CUE = 10_000  # in ms

# directory
SPRITES_DIR = "./sprites/"
LEVELS_DIR = "./levels/"
FROG_SOUND_FILEPATH = "./sounds/frog.mp3"

# sprite files
LILYPAD_FILE = 'lilypad.jpg'
LOG_FILE = 'log.png'
