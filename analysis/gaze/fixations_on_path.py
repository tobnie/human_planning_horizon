import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.data_utils import read_data


def check_on_path(time, fix_x, fix_y, path, radius):
    for t, x, y in path:
        x = int(x)
        y = int(y)
        if t > time:
            # TODO check in range around path field?
            if fix_x in range(x - radius, x + radius + 1, 1) and fix_y in range(y - radius, y + radius + 1, 1):
                return True
            # if x == fix_x and y == fix_y:
            #     return True

    return False


def fixation_ratio_on_path(game_df):
    path = game_df[['time', 'player_x_field', 'player_y_field']]
    path_list = list(path.itertuples(index=False, name=None))
    # TODO not only count fixations on field but also near path? Maybe use continuous fix-coords instead of field coords
    fixations = game_df[['time', 'fix_x_field', 'fix_y_field', 'fix_duration']]
    fixations['path_fix'] = fixations.apply(lambda x: check_on_path(x['time'], x['fix_x_field'], x['fix_y_field'], path_list, radius=1),
                                            axis=1)
    fixations.dropna(inplace=True)

    total_fix_duration = fixations['fix_duration'].sum()
    on_path_fix_duration = fixations[fixations['path_fix']]['fix_duration'].sum()

    on_path_ratio = on_path_fix_duration / total_fix_duration

    return on_path_ratio


def fixations_on_path():
    df = read_data()
    fix_df = pd.read_csv('../data/fixations.csv')

    df = df[['subject_id', 'game_difficulty', 'world_number', 'time', 'player_x_field', 'player_y_field']]
    df_with_fix = df.merge(fix_df, on=['subject_id', 'game_difficulty', 'world_number', 'time', 'player_x_field', 'player_y_field'],
                           how='left')

    trial_groups = df_with_fix.groupby(['subject_id', 'game_difficulty', 'world_number'])

    fixation_ratios = trial_groups.apply(fixation_ratio_on_path)
    print(fixation_ratios)
    fixation_ratios.to_csv('../data/on_path_fixation_ratios.csv')


def plot_fixation_ratios_over_scores():
    fixation_ratios = pd.read_csv('../data/on_path_fixation_ratios.csv')

    fixation_ratios = fixation_ratios.rename(columns={'0': 'fix_on_path'})

    level_scores = pd.read_csv('../data/level_scores.csv').drop_duplicates()
    game_durations = pd.read_csv('../data/game_durations.csv').drop_duplicates()
    fix_with_scores = fixation_ratios.merge(level_scores, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    fix_with_scores = fix_with_scores.merge(game_durations, on=['subject_id', 'game_difficulty', 'world_number'], how='left')

    sns.scatterplot(fix_with_scores[fix_with_scores['game_status'] == 'won'], x='standardized_level_score', y='fix_on_path',
                    hue='game_difficulty')
    plt.tight_layout()
    plt.show()

    sns.boxplot(fix_with_scores, x='game_status', y='fix_on_path')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # fixations_on_path()
    plot_fixation_ratios_over_scores()
