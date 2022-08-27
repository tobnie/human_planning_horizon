import numpy as np


def calc_sacc_amplitude(start_x, start_y, end_x, end_y):
    return np.sqrt((end_y - start_y) ** 2 + (end_x - start_x) ** 2)


def calc_sacc_angle(start_x, start_y, end_x, end_y):
    return np.arctan2(end_y - start_y, end_x - start_x)
