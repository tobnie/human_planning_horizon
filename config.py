# screen information
MONITOR_WIDTH_PX = 1600
MONITOR_HEIGHT_PX = 900    # TODO why is that?

# UI
font_size = 24

# world configuration
N_LANES = 21  # in previous thesis: 21
N_STREET_LANES = 9
N_WATER_LANES = 9

# obstacles
OBSTACLE_WIDTH = 100
OBSTACLE_WIDTH_MULTIPLIERS = [1, 2, 3]
OBSTACLE_SPAWN_TIME_MULTIPLIERS = [1, 2, 3]
PLAYER_VELOCITY = OBSTACLE_VELOCITY = 3.0

# game dynamics
# PLAYER_VELOCITY = 1.0   # (px) e.g. for moving
FPS = 60  # frame rate

# directory
SPRITES_DIR = "sprites"
