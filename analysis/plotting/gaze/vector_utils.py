import numpy as np


def calc_euclidean_distance(start_x, start_y, end_x, end_y):
    return np.sqrt((end_y - start_y) ** 2 + (end_x - start_x) ** 2)


def calc_manhattan_distance(start_x, start_y, end_x, end_y):
    return np.abs(int(end_y) - int(start_y)) + np.abs(int(end_x) - int(start_x))


def calc_angle(start_x, start_y, end_x, end_y):
    return np.arctan2(end_y - start_y, end_x - start_x)


def calc_angle_relative_to_vertical_center(start_x, start_y, end_x, end_y):
    angle_to_horizontal_center = calc_angle(start_x, start_y, end_x, end_y)
    return (angle_to_horizontal_center - np.pi / 2) / np.pi
