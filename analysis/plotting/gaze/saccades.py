import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.data_utils import read_data, get_street_data, get_river_data
from analysis.plotting.gaze.events.event_detection import try_saccade_detection
from analysis.plotting.gaze.vector_utils import calc_euclidean_distance, calc_angle

SACC_DURATION_THRESHOLD = 400


def get_saccade_dataframe(df):
    experience_df = df[['subject_id', 'game_difficulty', 'world_number', 'experience', 'trial', 'game_status', 'score']].drop_duplicates()
    data = df.groupby(['subject_id', 'game_difficulty', 'world_number'])[['subject_id', 'experience', 'time', 'gaze_x', 'gaze_y']].agg(
        {'time': list, 'gaze_x': list, 'gaze_y': list}).reset_index()
    data = data.merge(experience_df, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    sacc_df = data[
        ['subject_id', 'game_difficulty', 'world_number', 'gaze_x', 'gaze_y', 'time', 'experience', 'game_status', 'trial', 'score']].copy()
    sacc_df['sacc'] = sacc_df.apply(lambda x: try_saccade_detection(x['gaze_x'], x['gaze_y'], x['time'])[-1], axis=1)

    # remove rows where no saccades were detected
    sacc_df['sacc'] = sacc_df['sacc'].apply(lambda x: np.nan if len(x) == 0 else x)
    sacc_df.dropna(inplace=True)

    # reformat df
    sacc_df = sacc_df.explode('sacc')
    sacc_df.drop(['gaze_x', 'gaze_y', 'time'], axis=1, inplace=True)

    sacc_info_df = pd.DataFrame(sacc_df['sacc'].to_list(), columns=['time', 't_end', 'duration', 'x_start', 'y_start', 'x_end', 'y_end'])

    sacc_df.reset_index(inplace=True)
    sacc_df.drop(['index', 'sacc'], inplace=True, axis=1)
    sacc_df = pd.concat([sacc_df, sacc_info_df], axis=1)

    sacc_df['amplitude'] = sacc_df[['x_start', 'y_start', 'x_end', 'y_end']].apply(lambda x: calc_euclidean_distance(*x.values), axis=1)
    sacc_df['angle'] = sacc_df[['x_start', 'y_start', 'x_end', 'y_end']].apply(lambda x: calc_angle(*x.values), axis=1)

    sacc_df['sacc_x_end'] = sacc_info_df['x_end']
    sacc_df['sacc_y_end'] = sacc_info_df['y_end']
    sacc_df['duration'] = sacc_info_df['duration']
    return sacc_df


def filter_saccade_df(df):
    """ Filters the given dataframe by dropping saccades with a too long duration"""
    print('Saccades before Filtering:', len(df))
    filtered_df = df[df['duration'] <= SACC_DURATION_THRESHOLD]
    print('Saccades after Filtering:', len(filtered_df))
    return filtered_df


def plot_sacc_per_subject_experience(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='experience', y='amplitude', ax=ax)

    plt.savefig(directory + 'amplitudes_per_experience.png')
    plt.show()

    sns.barplot(data=df, x='experience', y='angle')
    plt.savefig(directory + 'angles_per_experience.png')
    plt.show()


def plot_sacc_per_subject_score(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)
    sns.barplot(data=df, y='amplitude', x='score', hue='experience')
    plt.title('the more red, the more experienced the player')
    plt.savefig(directory + 'amplitudes_per_score.png')
    plt.show()

    sns.barplot(data=df, y='angle', x='score', hue='experience')
    plt.title('the more red, the more experienced the player')
    plt.savefig(directory + 'angles_per_score.png')
    plt.show()


def plot_sacc_per_trial(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)
    # over all subjects
    sns.lineplot(data=df, y='amplitude', x='trial')
    plt.title('Sacc Amplitude per Trial')
    plt.savefig(directory + 'amplitudes_per_trial.png')
    plt.show()

    sns.lineplot(data=df, y='angle', x='trial')
    plt.title('Sacc Angle per Trial')
    plt.savefig(directory + 'angles_per_trial.png')
    plt.show()

    # per experience
    sns.lineplot(data=df, y='amplitude', x='trial', hue='experience')
    plt.title('Sacc Amplitude per Trial')
    plt.savefig(directory + 'amplitudes_per_trial_per_experience.png')
    plt.show()

    sns.lineplot(data=df, y='angle', x='trial', hue='experience')
    plt.title('Sacc Angle per Trial')
    plt.savefig(directory + 'angles_per_trial_per_experience.png')
    plt.show()

    # per subject
    sns.lineplot(data=df, y='amplitude', x='trial', hue='subject_id')
    plt.title('Sacc Amplitude per Trial')
    plt.savefig(directory + 'amplitudes_per_trial_per_subject.png')
    plt.show()

    sns.lineplot(data=df, y='angle', x='trial', hue='subject_id')
    plt.title('Sacc Angle per Trial')
    plt.savefig(directory + 'angles_per_trial_per_subject.png')
    plt.show()

    # per score
    g = sns.FacetGrid(df, col="score", margin_titles=True)
    g.map_dataframe(sns.barplot, x='trial', y='amplitude', ci=None)
    plt.suptitle('Sacc Amplitude per Trial')
    plt.tight_layout()
    plt.savefig(directory + 'amplitudes_per_trial_per_score.png')
    plt.show()

    g = sns.FacetGrid(df, col="score", margin_titles=True)
    g.map_dataframe(sns.barplot, x='trial', y='angle', ci=None)
    plt.suptitle('Sacc Angle per Trial')
    plt.tight_layout()
    plt.savefig(directory + 'angles_per_trial_per_score.png')
    plt.show()


def plot_sacc_per_difficulty(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)
    # over all subjects per difficulty
    sns.barplot(data=df, x='game_difficulty', y='amplitude')
    plt.title('Sacc Amplitude per Trial')
    plt.savefig(directory + 'amplitudes_per_difficulty.png')
    plt.show()

    sns.barplot(data=df, x='game_difficulty', y='angle')
    plt.title('Sacc Angle per Trial')
    plt.savefig(directory + 'angles_per_difficulty.png')
    plt.show()

    # over time per world
    g = sns.FacetGrid(df, col="world_number", row="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.lineplot, x='time', y='amplitude')
    plt.suptitle('Sacc Amplitude over time per world')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_over_time_per_world.png')
    plt.show()

    g = sns.FacetGrid(df, col="world_number", row="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.lineplot, x='time', y='angle')
    plt.suptitle('Sacc Angle over time per world')
    plt.tight_layout()
    plt.savefig(directory + 'angle_over_time_per_world.png')
    plt.show()

    # barplot over world
    g = sns.FacetGrid(df, col="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.barplot, x='world_number', y='amplitude', ci=None)
    plt.suptitle('Sacc Amplitude per world and difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_per_world_and_difficulty.png')
    plt.show()

    g = sns.FacetGrid(df, col="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.barplot, x='world_number', y='angle', ci=None)
    plt.suptitle('Sacc Angle per world and difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'angle_per_world_and_difficulty.png')
    plt.show()


def plot_sacc_per_game_status(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)
    # over all subjects per difficulty
    sns.barplot(data=df, x='game_status', y='amplitude')
    plt.title('Sacc Amplitude per game outcome')
    plt.savefig(directory + 'amplitudes_per_game_outcome.png')
    plt.show()

    sns.barplot(data=df, x='game_status', y='angle')
    plt.title('Sacc Angle per game outcome')
    plt.savefig(directory + 'angles_per_game_outcome.png')
    plt.show()

    # over time per world
    g = sns.FacetGrid(df, col="game_status", margin_titles=True)
    g.map_dataframe(sns.lineplot, x='time', y='amplitude')
    plt.suptitle('Sacc Amplitude over time per game outcome')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_over_time_per_game_outcome.png')
    plt.show()

    g = sns.FacetGrid(df, col="game_status", margin_titles=True)
    g.map_dataframe(sns.lineplot, x='time', y='angle')
    plt.suptitle('Sacc Angle over time per game outcome')
    plt.tight_layout()
    plt.savefig(directory + 'angle_over_time_per_game_outcome.png')
    plt.show()


def plot_sacc_histograms(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)

    # over game outcome
    g = sns.FacetGrid(df, col="game_status", margin_titles=True)
    g.map(sns.histplot, 'amplitude', stat='density')
    plt.suptitle('Sacc Amplitudes per game outcome')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_hist_per_game_outcome.png')
    plt.show()

    g = sns.FacetGrid(df, col="game_status", margin_titles=True)
    g.map(sns.histplot, 'angle', stat='density')
    plt.suptitle('Sacc Angles per game outcome')
    plt.tight_layout()
    plt.savefig(directory + 'angle_hist_per_game_outcome.png')
    plt.show()

    # over experience
    g = sns.FacetGrid(df, col="experience", margin_titles=True)
    g.map_dataframe(sns.histplot, 'amplitude', stat='density')
    plt.suptitle('Sacc Amplitudes per experience')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_hist_per_experience.png')
    plt.show()

    g = sns.FacetGrid(df, col="experience", margin_titles=True)
    g.map_dataframe(sns.histplot, 'angle', stat='density')
    plt.suptitle('Sacc Angles per experience')
    plt.tight_layout()
    plt.savefig(directory + 'angle_hist_per_experience.png')
    plt.show()

    # over difficulty
    g = sns.FacetGrid(df, col="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.histplot, 'amplitude', stat='density')
    plt.suptitle('Sacc Amplitudes per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_hist_per_difficulty.png')
    plt.show()

    g = sns.FacetGrid(df, col="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.histplot, 'angle', stat='density')
    plt.suptitle('Sacc Angles per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'angle_hist_per_difficulty.png')
    plt.show()

    # over score
    g = sns.FacetGrid(df, col="score", margin_titles=True)
    g.map_dataframe(sns.histplot, 'amplitude', stat='density')
    plt.suptitle('Sacc Amplitudes per Score')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_hist_per_score.png')
    plt.show()

    g = sns.FacetGrid(df, col="score", margin_titles=True)
    g.map_dataframe(sns.histplot, 'angle', stat='density')
    plt.suptitle('Sacc Angles per Score')
    plt.tight_layout()
    plt.savefig(directory + 'angle_hist_per_score.png')
    plt.show()

    # over difficulty
    g = sns.FacetGrid(df, col="world_number", row='game_difficulty', margin_titles=True)
    g.map_dataframe(sns.histplot, 'amplitude', stat='density')
    plt.suptitle('Sacc Amplitudes per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'amplitude_hist_per_difficulty_per_world.png')
    plt.show()

    g = sns.FacetGrid(df, col="world_number", row='game_difficulty', margin_titles=True)
    g.map_dataframe(sns.histplot, 'angle', stat='density')
    plt.suptitle('Sacc Angles per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'angle_hist_per_difficulty_per_world.png')
    plt.show()


def do_all_plots(df, subfolder=''):
    # get saccade data
    sacc_df = get_saccade_dataframe(df)

    # threshold saccades
    sacc_df = filter_saccade_df(sacc_df)

    plot_sacc_histograms(sacc_df, subfolder=subfolder)
    plot_sacc_per_game_status(sacc_df, subfolder=subfolder)
    plot_sacc_per_difficulty(sacc_df, subfolder=subfolder)
    plot_sacc_per_subject_score(sacc_df, subfolder=subfolder)
    plot_sacc_per_subject_experience(sacc_df, subfolder=subfolder)

    # need to drop nan trials for next plot:
    # drop rows with no trial information:
    df_without_nan_trials = df.dropna()
    sacc_df_without_nan_trials = get_saccade_dataframe(df_without_nan_trials)
    sacc_df_without_nan_trials = filter_saccade_df(sacc_df_without_nan_trials)

    plot_sacc_per_trial(sacc_df_without_nan_trials, subfolder=subfolder)


# load data
df = read_data()

do_all_plots(df)

street_df = get_street_data(df)
do_all_plots(street_df, subfolder='street/')

river_df = get_river_data(df)
do_all_plots(river_df, subfolder='river/')
