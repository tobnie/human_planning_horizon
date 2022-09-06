import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.data_utils import read_data, get_street_data, get_river_data, read_subject_data
from analysis.plotting.gaze.events.event_detection import try_saccade_detection, try_saccade_detection2
from analysis.plotting.gaze.vector_utils import calc_euclidean_distance, calc_angle, calc_angle_relative_to_vertical_center

SACC_DURATION_THRESHOLD = 400


def get_saccade_dataframe(df, use_other_saccade_algo=False):
    positions_df = df[['subject_id', 'game_difficulty', 'world_number', 'time', 'player_x_field', 'player_y_field']]
    experience_df = df[['subject_id', 'game_difficulty', 'world_number', 'experience', 'trial', 'game_status', 'score']].drop_duplicates()
    data = df.groupby(['subject_id', 'game_difficulty', 'world_number'])[
        ['subject_id', 'experience', 'time', 'gaze_x', 'gaze_y', 'player_y_field']].agg(
        {'time': list, 'gaze_x': list, 'gaze_y': list}).reset_index()
    data = data.merge(experience_df, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    sacc_df = data[
        ['subject_id', 'game_difficulty', 'world_number', 'gaze_x', 'gaze_y', 'time', 'experience', 'game_status', 'trial', 'score']].copy()

    if use_other_saccade_algo:
        # remove blinks for marius algo
        sacc_df['sacc'] = sacc_df.apply(lambda x: try_saccade_detection2(x['gaze_x'], x['gaze_y'], x['time'])[-1], axis=1)
    else:
        sacc_df['sacc'] = sacc_df.apply(lambda x: try_saccade_detection(x['gaze_x'], x['gaze_y'], x['time'])[-1], axis=1)

    # remove rows where no saccades were detected
    sacc_df['sacc'] = sacc_df['sacc'].apply(lambda x: np.nan if len(x) == 0 else x)
    sacc_df.dropna(subset=['sacc'], inplace=True)

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

    sacc_df = sacc_df.merge(positions_df, on=['subject_id', 'game_difficulty', 'world_number', 'time'], how='left')

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
    g.map(sns.histplot, 'amplitude', stat='density', binwidth=50)
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
    g.map_dataframe(sns.histplot, 'amplitude', stat='density', binwidth=50)
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
    g.map_dataframe(sns.histplot, 'amplitude', stat='density', binwidth=50)
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
    g.map_dataframe(sns.histplot, 'amplitude', stat='density', binwidth=50)
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
    g.map_dataframe(sns.histplot, 'amplitude', stat='density', binwidth=50)
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


def plot_sacc_boxplots(df, directory='./imgs/saccades/', subfolder=''):
    directory = directory + subfolder
    if not os.path.exists(directory):
        os.makedirs(directory)

    # over game outcome
    sns.boxplot(data=df, x='game_status', y='amplitude')
    plt.suptitle('Sacc Amplitudes per game outcome')
    plt.tight_layout()
    plt.savefig(directory + 'box_amplitude_per_game_outcome.png')
    plt.show()

    sns.boxplot(data=df, x='game_status', y='angle')
    plt.suptitle('Sacc Angles per game outcome')
    plt.tight_layout()
    plt.savefig(directory + 'box_angle_per_game_outcome.png')
    plt.show()

    # over experience
    sns.boxplot(data=df, x='experience', y='amplitude')
    plt.suptitle('Sacc Amplitudes per experience')
    plt.tight_layout()
    plt.savefig(directory + 'box_amplitude_per_experience.png')
    plt.show()

    sns.boxplot(data=df, x='experience', y='angle')
    plt.suptitle('Sacc Angles per experience')
    plt.tight_layout()
    plt.savefig(directory + 'box_angle_per_experience.png')
    plt.show()

    # over difficulty
    sns.boxplot(data=df, x='game_difficulty', y='amplitude')
    plt.suptitle('Sacc Amplitudes per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'box_amplitude_per_difficulty.png')
    plt.show()

    sns.boxplot(data=df, x='game_difficulty', y='angle')
    plt.suptitle('Sacc Angles per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'box_angle_per_difficulty.png')
    plt.show()

    # over difficulty
    g = sns.FacetGrid(data=df, row="game_difficulty", margin_titles=True)
    g.map_dataframe(sns.boxplot, 'world_number', 'amplitude')
    plt.suptitle('Sacc Amplitudes per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'box_amplitude_per_difficulty_per_world.png')
    plt.show()

    g = sns.FacetGrid(data=df, row='game_difficulty', margin_titles=True)
    g.map_dataframe(sns.boxplot, 'world_number', 'angle')
    plt.suptitle('Sacc Angles per difficulty')
    plt.tight_layout()
    plt.savefig(directory + 'box_angle_per_difficulty_per_world.png')
    plt.show()

    df_without_nan_score = df.dropna(subset=['score'])
    # over score
    sns.boxplot(data=df_without_nan_score, x='score', y='amplitude')
    plt.suptitle('Sacc Amplitudes per Score')
    plt.tight_layout()
    plt.savefig(directory + 'box_amplitude_per_score.png')
    plt.show()

    sns.boxplot(data=df_without_nan_score, x='score', y='angle')
    plt.suptitle('Sacc Angles per Score')
    plt.tight_layout()
    plt.savefig(directory + 'box_angle_per_score.png')
    plt.show()


def do_all_plots(sacc_df, subfolder=''):
    plot_sacc_boxplots(sacc_df, subfolder=subfolder)
    plot_sacc_histograms(sacc_df, subfolder=subfolder)
    plot_sacc_per_game_status(sacc_df, subfolder=subfolder)
    plot_sacc_per_difficulty(sacc_df, subfolder=subfolder)
    plot_sacc_per_subject_score(sacc_df, subfolder=subfolder)
    plot_sacc_per_subject_experience(sacc_df, subfolder=subfolder)

    # need to drop nan trials for next plot:
    # drop rows with no trial information:
    sacc_df_without_nan_trials = sacc_df.dropna(subset=['trial'])
    plot_sacc_per_trial(sacc_df_without_nan_trials, subfolder=subfolder)


def calc_visual_degree_from_px(x):
    d = 800  # mm
    w = 595  # mm
    h = 335  # mm
    res_x = 2560  # px
    res_y = 1440  # px
    mm_per_px_w = w / res_x
    mm_per_px_h = h / res_y
    mm_per_px = 0.5 * (mm_per_px_w + mm_per_px_h)
    deg = np.arctan(x * mm_per_px / d)
    return np.rad2deg(deg)


def sacc_duration_by_amplitude(amplitude_deg):
    """ Amplitude given in deg. Following the formula by Dodge and Cline (1901), Hyde (1959) and Robinson (1964).
    Only holds for amplitudes > 5°.
    """
    return 2.2 * amplitude_deg + 21  # ms


def filter_saccade_data2(df):
    df['amplitude_deg'] = df['amplitude'].apply(calc_visual_degree_from_px)
    df['sacc_duration_calculated'] = df['amplitude_deg'].apply(sacc_duration_by_amplitude)
    df['diff_in_calculation'] = df['duration'] - df['sacc_duration_calculated']
    sacc_df_only_over_5_deg = df[df['amplitude_deg'] > 5]

    # remove microsaccades (saccades under 5 minutes)
    df = df[df['amplitude_deg'] > 5 / 60]
    median_error = sacc_df_only_over_5_deg['diff_in_calculation'].median()

    # remove samples that deviate one median from the calculated duration
    df = df[df['diff_in_calculation'] <= median_error]
    return df


def print_per_score_mean_classification(df):
    df = df.dropna(subset=['score']).copy()
    print('-----Classification via over / under Mean-----')
    mean_score = df['score'].mean()
    df['scorer_type'] = df['score'].apply(lambda x: 'high' if x > mean_score else 'low')

    print(df[['subject_id', 'scorer_type']].drop_duplicates()['scorer_type'].value_counts())

    print('\n------By High / Low Scoring:')
    print('\nMean:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].mean())
    print('\nMedian:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].median())


def print_per_score_median_classification(df):
    df = df.dropna(subset=['score']).copy()
    print('-----Classification via over / under Median-----')
    median_score = df['score'].median()
    df['scorer_type'] = df['score'].apply(lambda x: 'high' if x > median_score else 'low').copy()

    print(df[['subject_id', 'scorer_type']].drop_duplicates()['scorer_type'].value_counts())

    print('\n------By High / Low Scoring:')
    print('\nMean:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].mean())
    print('\nMedian:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].median())
    print('\nVariance:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].var())

    sns.histplot(data=df, x='amplitude', hue='scorer_type')
    plt.show()


def print_per_score_half_classification(df):
    print('-----Classification via 50/50-----')

    df = df.dropna(subset=['score']).copy()

    # get middle value:
    subjects_with_score = df[['subject_id', 'score']].drop_duplicates()
    middle_score = subjects_with_score.sort_values(by='score')['score'].iloc[int(len(subjects_with_score) // 2)]

    df['scorer_type'] = df['score'].apply(lambda x: 'high' if x >= middle_score else 'low').copy()

    print(df[['subject_id', 'scorer_type']].drop_duplicates()['scorer_type'].value_counts())

    print('\n------By High / Low Scoring:')
    print('\nMean:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].mean())
    print('\nMedian:')
    print(df.groupby(['scorer_type'])[['amplitude', 'angle']].median())


def print_info(df):
    df['location'] = df['player_y_field'].apply(classify_location)

    print('------General:')
    print('\nMean:')
    print(df.groupby(['experience'])[['amplitude', 'angle']].mean())
    print('\nMedian:')
    print(df.groupby(['experience'])[['amplitude', 'angle']].median())

    print('\n------Per Subject:')
    print('\nMean:')
    print(df.groupby(['experience', 'subject_id'])[['amplitude', 'angle']].mean())
    print('\nMedian:')
    print(df.groupby(['experience', 'subject_id'])[['amplitude', 'angle']].median())

    print('------Per Location:')
    print('\nMean:')
    print(df.groupby(['location', 'experience'])[['amplitude', 'angle']].mean())
    print('\nMedian:')
    print(df.groupby(['location', 'experience'])[['amplitude', 'angle']].median())


def classify_location(player_y_field):
    if 8 <= player_y_field <= 13:
        return 'river'
    if 1 <= player_y_field <= 6:
        return 'street'
    if player_y_field == 0:
        return 'start'
    if player_y_field == 14:
        return 'finish'
    if player_y_field == 7:
        return 'middle'


def get_saccs_filter_and_plot(subfolder=''):
    # get saccade data
    # sacc_df = get_saccade_dataframe(df)
    # # threshold saccades
    # sacc_df = filter_saccade_df(sacc_df)
    # sacc_df = filter_saccade_data2(sacc_df)
    #
    # sacc_df.to_csv('saccades.csv')
    # print('Saved Saccade Information')
    sacc_df = pd.read_csv('saccades.csv')

    print(f'n={len(sacc_df)}')
    # print_per_score_half_classification(sacc_df)
    # print_per_score_mean_classification(sacc_df)
    print_per_score_median_classification(sacc_df)
    print_info(sacc_df)

    do_all_plots(sacc_df, subfolder=subfolder)

    # again for street
    street_df = get_street_data(sacc_df)
    sacc_df = get_saccade_dataframe(street_df)
    sacc_df = filter_saccade_df(sacc_df)
    sacc_df = filter_saccade_data2(sacc_df)
    do_all_plots(sacc_df, subfolder=subfolder + 'street/')

    # again for river
    river_df = get_river_data(sacc_df)
    sacc_df = get_saccade_dataframe(river_df)
    sacc_df = filter_saccade_df(sacc_df)
    sacc_df = filter_saccade_data2(sacc_df)
    do_all_plots(sacc_df, subfolder=subfolder + 'river/')


# load data
# df = read_data()

# other_algo_df = get_saccade_dataframe(df, use_other_saccade_algo=True)
# other_algo_df = filter_saccade_df(other_algo_df)
# other_algo_df, median_error_other_algo = filter_saccade_data2(other_algo_df)

subfolder = 'error_less_than_median_of_error/'
get_saccs_filter_and_plot(subfolder=subfolder)
