import config


class Field:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = config.MONITOR_WIDTH_PX / config.N_FIELDS_PER_LANE
        self.height = config.MONITOR_HEIGHT_PX / config.N_LANES
