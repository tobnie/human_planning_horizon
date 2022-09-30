import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.data_utils import read_data, subject2letter
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
    df.to_csv('../data/blinks.csv', index=False)
    print('Saved Blink Information')
    return df


def calc_ibi_per_game(game_df):
    blink_starts = game_df['blink_start']
    blink_ends = game_df['blink_end'].shift()
    ibis = (blink_starts - blink_ends).dropna()
    return ibis


def calculate_ibi():
    blink_df = pd.read_csv('../data/blinks.csv')
    blinks_per_game = blink_df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_ibi_per_game).reset_index(
        level=-1, drop=True).reset_index(name='ibi')
    return blinks_per_game


def save_blinks_and_IBIs():
    save_blinks()
    ibi_data = calculate_ibi()
    ibi_data.to_csv('../data/ibi.csv', index=False)


def plot_IBI_per_subject():
    ibi_data = pd.read_csv('../data/ibi.csv')
    level_scores = pd.read_csv('../data/level_scores.csv')[['subject_id', 'score']].drop_duplicates()

    ibi_data = ibi_data.merge(level_scores, on=['subject_id'], how='left')

    bin_width = 250
    ibi_data = ibi_data[ibi_data['ibi'] < 8_000]

    sns.histplot(data=ibi_data, x='ibi', binwidth=bin_width, stat='proportion')
    plt.savefig('./imgs/blinks/ibi/overall_ibi.png')
    plt.show()

    g = sns.FacetGrid(ibi_data, col='subject_id', col_wrap=4)
    g.map_dataframe(sns.histplot, 'ibi', binwidth=bin_width, stat='proportion')
    plt.savefig('./imgs/blinks/ibi/subject_ibi.png')
    plt.show()

    g = sns.FacetGrid(ibi_data, col='score', col_wrap=4)
    g.map_dataframe(sns.histplot, 'ibi', binwidth=bin_width, stat='proportion')
    plt.savefig('./imgs/blinks/ibi/score_ibi_hist.png')
    plt.show()

    order = ibi_data[['subject_id', 'score']].drop_duplicates().sort_values(by='score')
    ax = sns.boxplot(ibi_data, x='subject_id', y='ibi', order=order['subject_id'])

    # rename subject IDs
    xlabels = [subject2letter(subj_id.get_text()) for subj_id in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    plt.tight_layout()
    plt.savefig('./imgs/blinks/ibi/score_ibi_box.png')
    plt.show()


if __name__ == '__main__':
    # save_blinks_and_IBIs()
    plot_IBI_per_subject()
