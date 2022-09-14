import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.data_utils import read_data
from analysis.gaze.events.event_detection import try_blink_detection


def get_blink_df():
    df = read_data()

    experience_df = df[['subject_id', 'game_difficulty', 'world_number']].drop_duplicates()
    data = df.groupby(['subject_id', 'game_difficulty', 'world_number'])[['subject_id', 'experience', 'time', 'gaze_x', 'gaze_y']].agg(
        {'time': list, 'gaze_x': list, 'gaze_y': list}).reset_index()
    data = data.merge(experience_df, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    blink_df = data[['subject_id', 'game_difficulty', 'world_number', 'gaze_x', 'gaze_y', 'time']].copy()
    blink_df['blinks'] = blink_df.apply(lambda x: try_blink_detection(x['gaze_x'], x['gaze_y'], x['time'])[-1], axis=1)

    # remove rows where no saccades were detected
    blink_df['blinks'] = blink_df['blinks'].apply(lambda x: np.nan if len(x) == 0 else x)
    blink_df.dropna(subset=['blinks'], inplace=True)

    # reformat df
    blink_df = blink_df.explode('blinks')
    blink_df.drop(['gaze_x', 'gaze_y', 'time'], axis=1, inplace=True)

    blink_df_info = pd.DataFrame(blink_df['blinks'].to_list(), columns=['blink_start', 'blink_end', 'blink_duration'])

    blink_df.reset_index(inplace=True)
    blink_df.drop(['index', 'blinks'], inplace=True, axis=1)
    blink_df = pd.concat([blink_df, blink_df_info], axis=1)

    return blink_df


def save_blinks():
    df = get_blink_df()
    df.to_csv('blinks.csv', index=False)
    print('Saved Blink Information')
    return df


def calc_ibi_per_game(game_df):
    # blink_starts = game_df['blink_start'].iloc[1:].reset_index(drop=True)
    # blink_ends = game_df['blink_end'].iloc[:-1].reset_index(drop=True)
    # ibis = blink_starts - blink_ends

    blink_starts = game_df['blink_start']
    blink_ends = game_df['blink_end'].shift()
    ibis = (blink_starts - blink_ends).dropna()
    return ibis


def calculate_ibi():
    blink_df = pd.read_csv('blinks.csv')
    blinks_per_game = blink_df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_ibi_per_game).reset_index(level=-1,
                                                                                                                               drop=True).reset_index(
        name='ibi')
    return blinks_per_game


if __name__ == '__main__':
    # save_blinks()
    ibi_data = calculate_ibi()
    import seaborn as sns

    sns.histplot(data=ibi_data, x='ibi')
    plt.show()

    # TODO could not recreate ibi blink patterns
    sns.displot(data=ibi_data, x='ibi', col='subject_id', col_wrap=3)
    plt.show()
