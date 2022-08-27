import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.data_utils import read_subject_data, read_data, get_all_subjects
from analysis.plotting.gaze.events.event_detection import try_saccade_detection
from analysis.plotting.gaze.events.saccade_utils import calc_sacc_amplitude, calc_sacc_angle
from analysis.score_utils import add_max_score_to_df


def get_saccade_dataframe(df):
    experience_df = df[['subject_id', 'experience']].drop_duplicates()
    data = df.groupby(['subject_id', 'game_difficulty', 'world_number'])[['subject_id', 'experience', 'time', 'gaze_x', 'gaze_y']].agg(
        {'time': list, 'gaze_x': list, 'gaze_y': list}).reset_index()
    data = data.merge(experience_df, on=['subject_id'], how='left')
    sacc_df = data[['subject_id', 'gaze_x', 'gaze_y', 'time', 'experience']].copy()
    sacc_df['sacc'] = sacc_df.apply(lambda x: try_saccade_detection(x['gaze_x'], x['gaze_y'], x['time'])[-1], axis=1)

    # remove rows where no saccades were detected
    sacc_df['sacc'] = sacc_df['sacc'].apply(lambda x: np.nan if len(x) == 0 else x)
    sacc_df.dropna(inplace=True)

    # reformat df
    sacc_df = sacc_df.explode('sacc')
    sacc_df.drop(['gaze_x', 'gaze_y', 'time'], axis=1, inplace=True)

    sacc_info_df = pd.DataFrame(sacc_df['sacc'].to_list(), columns=['t_start', 't_end', 'duration', 'x_start', 'y_start', 'x_end', 'y_end'])

    sacc_df.reset_index(inplace=True)
    sacc_df.drop(['index', 'sacc'], inplace=True, axis=1)
    sacc_df = pd.concat([sacc_df, sacc_info_df], axis=1)

    sacc_df['amplitude'] = sacc_df[['x_start', 'y_start', 'x_end', 'y_end']].apply(lambda x: calc_sacc_amplitude(*x.values), axis=1)
    sacc_df['angle'] = sacc_df[['x_start', 'y_start', 'x_end', 'y_end']].apply(lambda x: calc_sacc_angle(*x.values), axis=1)
    return sacc_df


def plot_sacc_per_subject_experience(df):

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='experience', y='amplitude', ax=ax)

    plt.savefig('./imgs/saccades/amplitudes_per_experience.png')
    plt.show()

    sns.barplot(data=df, x='experience', y='angle')
    plt.savefig('./imgs/saccades/angles_per_experience.png')
    plt.show()


def plot_sacc_per_subject_score(df):

    sns.barplot(data=df, y='amplitude', x='score', hue='experience')
    plt.title('the more red, the more experienced the player')
    plt.savefig('./imgs/saccades/amplitudes_per_score.png')
    plt.show()

    sns.barplot(data=df, y='angle', x='score', hue='experience')
    plt.title('the more red, the more experienced the player')
    plt.savefig('./imgs/saccades/angles_per_score.png')
    plt.show()



df = read_data()
# df = read_subject_data('JO03SA')
sacc_df = get_saccade_dataframe(df)

# add max score to sacc_df
sacc_df = add_max_score_to_df(sacc_df)

plot_sacc_per_subject_score(sacc_df)
plot_sacc_per_subject_experience(sacc_df)
