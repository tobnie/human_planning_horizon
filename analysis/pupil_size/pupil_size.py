import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from analysis.data_utils import read_data


def get_nan_idx(df, expand=2):
    pupil_sizes = df['pupil_size']
    idx = list(df[pupil_sizes == 0].index)

    # return no indices if there are
    if len(idx) == 0:
        return []

    # expand indices by 2 in each direction to also drop samples near a blink
    prev_idx = idx[0]
    expanded_idx = [prev_idx]

    # add extra samples before first index
    for i in range(expand):
        shift = i + 1
        if prev_idx - shift >= pupil_sizes.index.min():
            expanded_idx.append(prev_idx - shift)

    for i in idx[1:]:
        expanded_idx.append(i)

        if i - prev_idx > 1:
            expanded_idx.append(prev_idx + 1)
            expanded_idx.append(prev_idx + 2)
            expanded_idx.append(i - 2)
            expanded_idx.append(i - 1)

        # iterate
        prev_idx = i

    # add extra samples after last index
    last_index = idx[-1]
    for i in range(expand):
        shift = i + 1
        if last_index + shift < pupil_sizes.index.max():
            expanded_idx.append(last_index + shift)

    return expanded_idx


def set_gaze_information_nan(game_df):
    expand_around_na_samples = 2
    idx = get_nan_idx(game_df, expand_around_na_samples)
    if len(idx) > 0:
        game_df['pupil_size'][idx] = np.nan
        # game_df['gaze_x'][idx] = -32768 # TODO do that or not?
        # game_df['gaze_y'][idx] = -32768
    return game_df


def add_pupil_size_z_score(subject_df):
    # drop zero pupil size and samples around them
    subject_df = subject_df.groupby(['subject_id', 'game_difficulty', 'world_number'], group_keys=False).apply(set_gaze_information_nan).reset_index()

    # standardize for subject
    pupil_size = subject_df['pupil_size']
    subject_df['pupil_size_z'] = (pupil_size - pupil_size.mean()) / pupil_size.std()
    return subject_df


def plot_pupil_size_over_time_per_game(game_df):
    game_df.dropna(subset='pupil_size_z', inplace=True)
    subject_id = game_df['subject_id'].values[0]
    game_difficulty = game_df['game_difficulty'].values[0]
    world_number = game_df['world_number'].values[0]
    plt.plot(game_df['time'], game_df['pupil_size_z'])
    file_name = 'pupil_size_{}_{}_{}.png'.format(subject_id, game_difficulty, world_number)
    plt.title(file_name)
    plt.xlabel('time [ms]')
    plt.ylabel('pupil size [mm^2], z-standardized')
    plt.savefig('./imgs/pupil_size/per_subject_over_time/' + file_name)
    plt.show()


def plot_pupil_size_for_each_game():
    df = read_data()
    df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(plot_pupil_size_over_time_per_game)


def plot_pupil_size():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')

    # pupil size everywhere
    pupil_size_total = df['pupil_size_z']
    print('--- Pupil Size Total ---')
    print('Mean: ', pupil_size_total.mean())
    print('Median: ', pupil_size_total.median())
    print('Variance: ', pupil_size_total.var())

    # pupil size on street
    street_mask = df['region'] == 'street'
    pupil_size_street = df[street_mask]['pupil_size_z']

    print('--- Street ---')
    print('Mean: ', pupil_size_street.mean())
    print('Median: ', pupil_size_street.median())
    print('Variance: ', pupil_size_street.var())

    # pupil size on river
    river_mask = df['region'] == 'river'
    pupil_size_river = df[river_mask]['pupil_size_z']
    print('--- River ---')
    print('Mean: ', pupil_size_river.mean())
    print('Median: ', pupil_size_river.median())
    print('Variance: ', pupil_size_river.var())

    plt.boxplot([pupil_size_total, pupil_size_street, pupil_size_river], labels=['total', 'street', 'river'])
    plt.ylabel('pupil size z-score')
    plt.savefig('./imgs/pupil_size/pupil_sizes_box.png')
    plt.show()

    # TODO position names
    plt.violinplot(dataset=[pupil_size_total, pupil_size_street, pupil_size_river],
                   showextrema=False)  # , positions=['total', 'street', 'river'])
    plt.ylabel('pupil size z-score')
    plt.ylim((-3, 5))
    plt.savefig('./imgs/pupil_size/pupil_sizes_violin.png')
    plt.show()


def ttest_pupil_size_street_river():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')

    # pupil size on street
    street_mask = df['region'] == 'street'
    pupil_size_street = df[street_mask]['pupil_size_z']

    # pupil size on river
    river_mask = df['region'] == 'river'
    pupil_size_river = df[river_mask]['pupil_size_z']

    # t test var 1
    print('H0: Pupil sizes are equal | H1: Pupil sizes on street are greater than pupil sizes on river')
    ttest_result = scipy.stats.ttest_ind(pupil_size_street, pupil_size_river, alternative='greater')
    print(ttest_result)
    print('dof=', len(pupil_size_street) - 1 + len(pupil_size_river) - 1)


def kstest_pupil_size_street_river():
    df = read_data()
    df = df[['region', 'pupil_size_z']].dropna(subset='pupil_size_z')

    # pupil size on street
    street_mask = df['region'] == 'street'
    pupil_size_street = df[street_mask]['pupil_size_z']

    # pupil size on river
    river_mask = df['region'] == 'river'
    pupil_size_river = df[river_mask]['pupil_size_z']
    print('H0: Distributions for pupil sizes are equal | H1: Distributions for pupil sizes are different for street and river')

    # kstest
    kstest_result = scipy.stats.kstest(pupil_size_river, pupil_size_street, alternative='two-sided')
    print(kstest_result)


if __name__ == '__main__':
    plot_pupil_size()
    ttest_pupil_size_street_river()
    kstest_pupil_size_street_river()
    plot_pupil_size_for_each_game()
