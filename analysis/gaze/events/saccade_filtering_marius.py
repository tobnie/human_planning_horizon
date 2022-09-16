import numpy as np
import pandas as pd

HORIZONTAL_RES = 2560
VERTICAL_RES = 1440
SCREEN_WIDTH_MM = 595 # TODO maybe it only did not work because of switched values here
SCREEN_HEIGHT_MM = 335
D = 800


def x_to_mm(x):
    return x * (SCREEN_WIDTH_MM / HORIZONTAL_RES)


def y_to_mm(y):
    return y * (SCREEN_HEIGHT_MM / VERTICAL_RES)


def calc_angle(a_x, a_y, e_x, e_y):
    a_e = np.sqrt(
        x_to_mm(a_x - e_x) ** 2 + y_to_mm(a_y - e_y, ) ** 2)
    a = np.sqrt(x_to_mm(a_x) ** 2 + y_to_mm(a_y, ) ** 2)
    e = np.sqrt(x_to_mm(e_x) ** 2 + y_to_mm(e_y, ) ** 2)
    alpha_2 = np.arctan(a / D)
    alpha_1 = np.arctan(e / D)
    d_a = D / np.cos(alpha_2)
    d_e = D / np.cos(alpha_1)
    alpha = np.arccos((d_a ** 2 + d_e ** 2 - a_e ** 2) / (2 * d_a * d_e))

    return a_e, np.degrees(alpha_1), np.degrees(alpha), np.degrees(alpha_2)


def detect_samples_marius(sample_list, main_speed_threshold=240, main_acc_threshold=3000, secondary_threshold=25,
                          use_secondary_threshold=False):
    mm_dist = [np.nan]
    curr_angle = []
    ang_dist = [np.nan]
    ang_speed = [np.nan]
    ang_acc = [np.nan]
    time_passed = [np.nan]
    prev_time, x1, y1 = sample_list[0]
    xs = [x1]
    ys = [y1]
    for i in range(1, len(sample_list)):
        time, x2, y2 = sample_list[i]

        mm, point_angle, view_angle, prev_point_angle = calc_angle(x1, y1, x2, y2)

        if i == 1:
            curr_angle.append(prev_point_angle)
        mm_dist.append(mm)
        curr_angle.append(point_angle)
        ang_dist.append(view_angle)
        delta_t = time - prev_time

        time_factor = 1 / delta_t
        ang_speed.append(view_angle * time_factor)
        delta_speed = ang_speed[-1] - ang_speed[-2]
        ang_acc.append(delta_speed * time_factor)
        time_passed.append(delta_t)

        xs.append(x2)
        ys.append(y2)

        # set iterating variables
        prev_time = time
        x1 = x2
        y1 = y2

    df_dict = {'x': xs, 'y': ys, 'dist': mm_dist, 'curr_angle': curr_angle, 'ang_dist': ang_dist, 'ang_speed': ang_speed, 'ang_acc': ang_acc}
    angles_df = pd.DataFrame(df_dict)
    angles_df["saccade"] = ((angles_df.ang_speed >= main_speed_threshold) | (angles_df.ang_acc >= main_acc_threshold)) * 1

    if use_secondary_threshold:
        # marking everything consecutively after an identified saccade that meets the secondary velocity threshold as saccade
        for i in angles_df.index:
            if angles_df.iloc[i - 1].saccade != 0 and angles_df.iloc[i].ang_speed >= secondary_threshold:
                # not overwriting main saccades
                if angles_df.loc[i, "saccade"] == 0:
                    angles_df.loc[i, "saccade"] = 2

        # reverse order to do the same thing for consecutive timestamps before a main-thresh saccade
        for i in reversed(angles_df.index):
            if angles_df.iloc[i - 1].saccade != 0 and angles_df.iloc[i].ang_speed >= secondary_threshold:
                # not overwriting main saccades
                if angles_df.loc[i, "saccade"] == 0:
                    angles_df.loc[i, "saccade"] = 2

    return angles_df
