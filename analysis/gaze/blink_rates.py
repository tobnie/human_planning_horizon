import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from analysis.data_utils import get_river_data, get_street_data, read_data


def calc_blink_rate_per_game(game_df):
    game_df['time_delta'] = game_df['time'].diff()

    street_df = get_street_data(game_df)
    river_df = get_river_data(game_df)

    # game time and times on areas
    # remember that game_time >= time_on_street + time_on_river since there are also start and middle lane
    # also convert from msec to sec
    game_time = game_df['time'].max() * 1e-3
    time_on_street = street_df['time_delta'].sum() * 1e-3
    time_on_river = river_df['time_delta'].sum() * 1e-3

    # number of blinks
    n_blinks_street = len(street_df[street_df['blink_start'].notna()])
    n_blinks_river = len(river_df[river_df['blink_start'].notna()])
    n_blinks_total = len(game_df[game_df['blink_start'].notna()])

    # calculate blink rates
    blink_rate_street = n_blinks_street / time_on_street if time_on_street > 0 else np.nan
    blink_rate_river = n_blinks_river / time_on_river if time_on_river > 0 else np.nan
    blink_rate_total = n_blinks_total / game_time if n_blinks_total > 0 else np.nan

    return blink_rate_total, blink_rate_street, blink_rate_river


def save_blink_rates():
    blinks = pd.read_csv('../data/blinks.csv')
    df = read_data()

    joined_df = df.merge(blinks, left_on=['subject_id', 'game_difficulty', 'world_number', 'time'],
                         right_on=['subject_id', 'game_difficulty', 'world_number', 'blink_start'], how='left')
    blink_rate_info = joined_df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_blink_rate_per_game).reset_index(
        name='blink_info')
    blink_info = pd.DataFrame(blink_rate_info['blink_info'].to_list(), columns=['blink_rate', 'blink_rate_street', 'blink_rate_river'])
    blink_rates = pd.concat([blink_rate_info[['subject_id', 'game_difficulty', 'world_number']], blink_info], axis=1)
    blink_rates.to_csv('../data/blink_rates.csv', index=False)
    return blink_rates


def plot_blink_rates():
    blink_rates = pd.read_csv('../data/blink_rates.csv')

    blink_rates_total = blink_rates['blink_rate'].dropna()
    blink_rates_street = blink_rates['blink_rate_street'].dropna()
    blink_rates_river = blink_rates['blink_rate_river'].dropna()

    # blink rates everywhere
    print('--- Blink Rate Total ---')
    print('Mean: ', blink_rates_total.mean())
    print('Median: ', blink_rates_total.median())
    print('Variance: ', blink_rates_total.var())

    # blink rates on street
    print('--- Street ---')
    print('Mean: ', blink_rates_street.mean())
    print('Median: ', blink_rates_street.median())
    print('Variance: ', blink_rates_street.var())

    # blink rates on river
    print('--- River ---')
    print('Mean: ', blink_rates_river.mean())
    print('Median: ', blink_rates_river.median())
    print('Variance: ', blink_rates_river.var())

    plt.boxplot([blink_rates_total.values, blink_rates_street.values, blink_rates_river.values], labels=['total', 'street', 'river'])
    plt.ylabel('blink rate [blinks/s]')
    plt.savefig('./imgs/blinks/blink_rates_box.png')
    plt.show()


def ttest_blink_rate_street_river():
    blink_rates = pd.read_csv('../data/blink_rates.csv')
    blink_rates_street = blink_rates['blink_rate_street'].dropna().values
    blink_rates_river = blink_rates['blink_rate_river'].dropna().values

    print('H0: Blink rates are equal | H1: Blink rates on river are greater than blink rates on street')

    # perform (Welch's) t-test
    # t test euclidean distances:
    ttest_result = scipy.stats.ttest_ind(blink_rates_river, blink_rates_street,  alternative='greater')
    print(ttest_result)
    print('dof=', len(blink_rates_street) - 1 + len(blink_rates_river) - 1)


def kstest_blink_rate_distance_street_river():
    blink_rates = pd.read_csv('../data/blink_rates.csv')
    blink_rates_street = blink_rates['blink_rate_street'].dropna().values
    blink_rates_river = blink_rates['blink_rate_river'].dropna().values

    print('H0: Distributions for blink rates are equal | H1: Distributions for blink rates are different for street and river')

    # perform (Welch's) t-test
    # t test euclidean distances:
    kstest_result = scipy.stats.kstest(blink_rates_river, blink_rates_street, alternative='two-sided')
    print(kstest_result)


if __name__ == '__main__':
    save_blink_rates()
    # plot_blink_rates()
    # ttest_blink_rate_street_river()
    # kstest_blink_rate_distance_street_river()
