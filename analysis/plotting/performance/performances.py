import itertools

import pandas as pd
from matplotlib import pyplot as plt

from analysis.analysis_utils import get_all_subjects
from analysis.data_utils import read_data
from analysis.performance.games_won import get_won_games, get_last_time_steps_of_games, WIN_THRESHOLD_Y, TIME_OUT_THRESHOLD
import seaborn as sns


def return_status(time, player_y):
    if player_y >= WIN_THRESHOLD_Y:
        return 'won'
    elif time >= TIME_OUT_THRESHOLD:
        return 'timed_out'
    else:
        return 'lost'


def add_game_status_to_df(df):
    last_time_steps = get_last_time_steps_of_games(df).copy()
    last_time_steps['game_status'] = last_time_steps.apply(lambda x: return_status(x['time'], x['player_y']), axis=1)
    df_filtered = last_time_steps[['subject_id', 'game_difficulty', 'game_status']]
    df = df.merge(df_filtered, on=['subject_id', 'game_difficulty'])
    return df


def plot_performance_per_difficulty(df=None):
    if df is None:
        df = read_data()

    last_time_steps = get_last_time_steps_of_games(df).copy()

    last_time_steps['game_status'] = last_time_steps.apply(lambda x: return_status(x['time'], x['player_y']), axis=1)

    # counts = df.groupby(['subject_id', 'game_difficulty', 'game_status']).tails(1).size()
    counts = last_time_steps.groupby(['subject_id', 'game_difficulty', 'game_status']).size().reset_index().rename(columns={0: 'count'})

    combined = [get_all_subjects(), ['easy', 'normal', 'hard'], ['won', 'timed_out', 'lost']]
    all_combinations = pd.DataFrame(columns=['subject_id', 'game_difficulty', 'game_status'], data=list(itertools.product(*combined)))

    counts = all_combinations.merge(counts, on=['subject_id', 'game_difficulty', 'game_status'], how='left').fillna(0)

    print(counts)
    # plot number of game outcomes
    g = sns.catplot(x="game_difficulty", hue="game_status", col="subject_id", y='count', col_wrap=4, kind='bar', data=counts, height=4,
                    aspect=.7)
    plt.savefig('./imgs/performance/game_endings_per_subject.png')
    plt.show()

    # plot number of game outcomes per difficulty
    g = sns.catplot(x="game_difficulty", hue="game_status", y='count', kind='bar', data=counts, height=4,
                    aspect=.7)
    plt.savefig('./imgs/performance/game_endings.png')
    plt.show()

    # outcomes as stacked bar plot
    last_time_steps.groupby(['game_difficulty', 'game_status'])['game_difficulty'].count().unstack('game_status').fillna(0).plot(kind='bar',
                                                                                                         stacked=True)
    plt.savefig('./imgs/performance/game_endings_stacked.png')

    # plot average time
    g = sns.catplot(x="game_difficulty", hue="game_status", y='time', kind='bar', data=last_time_steps, height=4,
                    aspect=.7)
    plt.savefig('./imgs/performance/game_times.png')
    plt.show()


# plot_performance_per_difficulty()
