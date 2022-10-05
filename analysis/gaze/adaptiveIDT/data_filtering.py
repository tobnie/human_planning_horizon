from math import atan2, degrees
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from analysis.data_utils import read_subject_data
from analysis.plotting_utils import color_fader

SAMPLING_FREQUENCY = 60  # Hz
DELTA_T = 1 / SAMPLING_FREQUENCY

HORIZONTAL_RES = 2560
VERTICAL_RES = 1440
SCREEN_WIDTH_MM = 595
SCREEN_HEIGHT_MM = 335
D = 1020

# horizontally
PIX2DEG_HORIZ = degrees(atan2(SCREEN_WIDTH_MM / 2, D)) / (HORIZONTAL_RES / 2)
# vertically
PIX2DEG_VERT = degrees(atan2(SCREEN_HEIGHT_MM / 2, D)) / (VERTICAL_RES / 2)
PHI = PIX2DEG = (PIX2DEG_HORIZ + PIX2DEG_VERT) / 2


def savitzky_golay_filter(gaze_x, gaze_y):
    """
    Performs Savitzky Golay Filtering with a window length of 2 (~33ms) with a polynomial order of 2. Calculates angular velocities and
    accelerations from the smoothed samples and returns these.
    :param gaze_x: x coordinates of 3gaze samples
    :param gaze_y: y coordinates of 3gaze samples
    :return: (angular velocities, angular accelerations)
    """
    # TODO how to choose. window would be next to 20ms with only 1 or 2 but that does not make a lot of sense. Anyway choose higher?
    window_length = 23
    poly_order = 2

    miss_indices = get_indices_of_missing_data(gaze_x, gaze_y)
    gaze_x = gaze_x[~miss_indices]
    gaze_y = gaze_y[~miss_indices]

    # filter for x
    x = savgol_filter(gaze_x, window_length=window_length, polyorder=poly_order)
    dx = np.abs(savgol_filter(gaze_x, window_length=window_length, polyorder=poly_order, deriv=1, delta=1 / SAMPLING_FREQUENCY))
    ddx = np.abs(savgol_filter(gaze_x, window_length=window_length, polyorder=poly_order, deriv=2, delta=1 / SAMPLING_FREQUENCY))

    # finite differences x
    dx_finite = np.abs(np.diff(gaze_x) / DELTA_T)

    # TODO set to zero, remove afterwards, ...?
    # dx_finite[miss_indices[1:]] = 0
    # dx[miss_indices] = 0

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(dx_finite)) * DELTA_T, dx_finite)
    ax.plot(np.arange(len(dx)) * DELTA_T, dx, linestyle='--')
    ax.set_title('dx')
    plt.show()

    # filter for y
    y = savgol_filter(gaze_y, window_length=window_length, polyorder=poly_order)
    dy = np.abs(savgol_filter(gaze_y, window_length=window_length, polyorder=poly_order, deriv=1, delta=1 / SAMPLING_FREQUENCY))
    ddy = np.abs(savgol_filter(gaze_y, window_length=window_length, polyorder=poly_order, deriv=2, delta=1 / SAMPLING_FREQUENCY))

    dtheta = SAMPLING_FREQUENCY * PHI * np.sqrt(dx ** 2 + dy ** 2)
    ddtheta = SAMPLING_FREQUENCY * PHI * np.sqrt(ddx ** 2 + ddy ** 2)

    PT = find_peak_velocity_threshold(dtheta)
    print(PT)

    return dtheta, ddtheta


def get_indices_of_missing_data(x, y, missing=-32768):
    mx = np.array(x == missing, dtype=int)
    my = np.array(y == missing, dtype=int)
    miss = np.array((mx + my) == 2, dtype=bool)
    return miss


def find_peak_velocity_threshold(dtheta, initial_PT=200):
    PT_prev = np.inf
    PT = initial_PT

    PTs = []

    while np.abs(PT_prev - PT) >= 1:
        PTs.append(PT)
        # get samples below threshold
        samples_below_threshold = dtheta[dtheta < PT]
        mu = samples_below_threshold.mean()
        std = samples_below_threshold.std()
        PT_prev = PT
        PT = mu + 6 * std

    PTs.append(PT)

    fig, ax = plt.subplots()

    ax.plot(dtheta)
    for i, pt in enumerate(PTs):
        mix = i / len(PTs)
        color = color_fader('yellow', 'red', mix)
        ax.axhline(pt, linestyle='--', color=color)

    plt.show()

    return PT


if __name__ == '__main__':
    df = read_subject_data('ED06RA')
    trial_df = df[df['trial'] == 35]
    x, y = trial_df['gaze_x'].to_numpy(), trial_df['gaze_y'].to_numpy()
    savitzky_golay_filter(x, y)
