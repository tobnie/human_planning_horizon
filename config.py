############################################### DISPLAY ################################################
DISPLAY_WIDTH_PX = 1920# 2560  # pygame.display.Info().current_w
DISPLAY_HEIGHT_PX = 1080 #1440  # pygame.display.Info().current_h

############################################### UI ################################################
font_size = 24

############################################### WORLD ################################################
N_LANES = 15
N_STREET_LANES = 6
N_WATER_LANES = 6

N_FIELDS_PER_LANE = 20  # in original game 14 I guess
FIELD_WIDTH = DISPLAY_WIDTH_PX / N_FIELDS_PER_LANE
FIELD_HEIGHT = ROW_HEIGHT = DISPLAY_HEIGHT_PX / N_LANES

PLAYER_WIDTH_TO_FIELD_WIDTH_RATIO = .65
PLAYER_WIDTH = PLAYER_WIDTH_TO_FIELD_WIDTH_RATIO * FIELD_WIDTH
PLAYER_HEIGHT_TO_FIELD_HEIGHT_RATIO = .65
PLAYER_HEIGHT = PLAYER_HEIGHT_TO_FIELD_HEIGHT_RATIO * FIELD_HEIGHT

############################################### PLAYER ################################################
PLAYER_MOVEMENT_BOUNDS_X = (-FIELD_WIDTH, DISPLAY_WIDTH_PX)
PLAYER_MOVEMENT_BOUNDS_Y = (0, DISPLAY_HEIGHT_PX - FIELD_HEIGHT)

############################################ GAME DYNAMICS ############################################
FPS = 60  # frame rate
LEVEL_TIME = 75_000  # in ms
LEVEL_TIME_AUDIO_CUE = 10_000  # in ms
DELAY_AFTER_LEVEL_FINISH = 1_000  # in ms (time until game is closed after level is finished)
PRE_RUN_ITERATIONS = 2000

# obstacles
PLAYER_UPDATE_INTERVAL = 350

################################################ SCORE ################################################
# points
WIN_BONUS = 200
DEATH_PENALTY = -100
VISITED_LANE_BONUS = 5

# point multiplier
EASY_MULTIPLIER = 1
NORMAL_MULTIPLIER = 2
HARD_MULTIPLIER = 3

################################################ FILES ################################################
# directory
SPRITES_DIR = "game/sprites/"
LEVELS_DIR = "game/levels/"
LOG_DIR = "./data/"
LEVEL_DATA_DIR = LOG_DIR + "level_data/"
SCORE_DIR = LOG_DIR + "scores/"
FROG_SOUND_FILEPATH = "./sounds/frog.mp3"

# sprite files
LILYPAD_FILE = 'lilypad.png'
LOG_FILE = 'plank.jpg'
CAR_FILE = 'car.png'
STAR_FILE = 'star.png'

############################################### EYE TRACKER ################################################
MISSING_SAMPLE = -32768
