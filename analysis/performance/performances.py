import itertools
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from analysis import paper_plot_utils
from analysis.data_utils import read_data, get_all_subjects, get_last_time_steps_of_games, subject2letter
import seaborn as sns

from analysis.score.recalculate_score import add_level_score_estimations_to_df


def plot_performance_per_difficulty():
    last_time_steps = get_last_time_steps_of_games().copy()
    counts = pd.read_csv('../data/performance_stats.csv', index_col=0)

    # plot number of game outcomes
    g = sns.catplot(x="game_difficulty", hue="game_status", col="subject_id", y='percentage', col_wrap=4, kind='bar', data=counts, height=4,
                    aspect=.7)
    g.set(ylim=(0.0, 1.0))
    plt.savefig('./imgs/performance/game_endings_per_subject_per_difficulty.png')
    plt.show()

    g = sns.catplot(x="game_status", col="subject_id", y='percentage', col_wrap=4, kind='bar', data=counts, height=4,
                    aspect=.7)
    g.set(ylim=(0.0, 1.0))
    plt.savefig('./imgs/performance/game_endings_per_subject.png')
    plt.show()

    # plot number of game outcomes per difficulty
    g = sns.catplot(x="game_difficulty", hue="game_status", y='percentage', kind='bar', data=counts, height=4,
                    aspect=.7)
    g.set(ylim=(0.0, 1.0))
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


def get_absolute_counts(df):
    df['won_total'] = df[df['game_status'] == 'won'].sum()
    df['won_total_percentage'] = df['won_total'] / 60
    return df


def save_performance_stats():
    df = read_data()

    last_time_steps = get_last_time_steps_of_games(df).copy()

    # counts = df.groupby(['subject_id', 'game_difficulty', 'game_status']).tails(1).size()
    counts = last_time_steps.groupby(['subject_id', 'game_difficulty', 'game_status']).size().reset_index().rename(columns={0: 'count'})

    combined = [get_all_subjects(), ['easy', 'normal', 'hard'], ['won', 'timed_out', 'lost']]
    all_combinations = pd.DataFrame(columns=['subject_id', 'game_difficulty', 'game_status'], data=list(itertools.product(*combined)))

    counts = all_combinations.merge(counts, on=['subject_id', 'game_difficulty', 'game_status'], how='left').fillna(0)
    counts_per_difficulty = counts.merge(last_time_steps[['subject_id', 'score']], on='subject_id', how='left').drop_duplicates()

    # normalized_counts PER DIFFICULTY
    games_per_difficulty = 20
    counts_per_difficulty['percentage'] = counts_per_difficulty['count'].div(games_per_difficulty)

    counts_per_difficulty.to_csv('performance_stats.csv', index=False)
    print('Saved Performance Stats')


def print_average_game_endings():
    df = pd.read_csv('../data/performance_stats.csv', index_col=0)

    # df = df.apply(lambda x: get_absolute_counts(df)).reset_index()

    wins = df.groupby(['subject_id', 'game_status']).agg(['mean', 'std', 'median']).reset_index()

    # all_wins = df['game_status'].agg()

    print('Average Won Games: ', df[df['game_status'] == 'won']['percentage'].mean())
    print('Average Timed Out Games: ', df[df['game_status'] == 'timed_out']['percentage'].mean())
    print('Average Lost Games: ', df[df['game_status'] == 'lost']['percentage'].mean())

    print('----- Per difficulty -----')
    print('Average Won Games: ', df[df['game_status'] == 'won'].groupby('game_difficulty')['percentage'].mean())
    print('Average Timed Out Games: ', df[df['game_status'] == 'timed_out'].groupby('game_difficulty')['percentage'].mean())
    print('Average Lost Games: ', df[df['game_status'] == 'lost'].groupby('game_difficulty')['percentage'].mean())

    # best subject:
    scores = df[['subject_id', 'percentage', 'game_status', 'score']].groupby(['subject_id', 'game_status', 'score']).agg(
        percentage_over_all=('percentage', 'mean')).reset_index()
    best_subject = scores.loc[scores['score'] == scores['score'].max()]
    print('\n\n ---- Best Subject ----')
    print('Won Games: ', best_subject[best_subject['game_status'] == 'won']['percentage_over_all'].mean())

    # worst subject
    worst_subject = scores.loc[scores['score'] == scores['score'].min()]
    print('\n\n ---- Worst Subject ----')
    print('Won Games: ', worst_subject[worst_subject['game_status'] == 'won']['percentage_over_all'].mean())

    # Variance and median of won games:
    print("\n\n")
    print('Variance of won games: ', scores[scores['game_status'] == 'won']['percentage_over_all'].var())
    print('Median of won games: ', scores[scores['game_status'] == 'won']['percentage_over_all'].median())
    # print(df)


def save_game_durations():
    last_time_steps = get_last_time_steps_of_games().copy()
    durations = last_time_steps[['subject_id', 'game_difficulty', 'world_number', 'time', 'score', 'experience', 'game_status']]
    durations.to_csv('game_durations.csv', index=False)
    print('Saved Game Durations')


def histogram_over_avg_trial_times():
    game_durations = pd.read_csv('../data/game_durations.csv')

    game_durations_won = game_durations[game_durations['game_status'] == 'won'].copy()

    game_durations_won.loc[:, 'time'] = game_durations_won.loc[:, 'time'].div(1000)

    print('Completion Time Mean: ', game_durations_won['time'].mean())
    print('Completion Time Var: ', game_durations_won['time'].var())
    print('Completion Time Median: ', game_durations_won['time'].median())

    game_durations_won.replace('normal', 'medium', inplace=True)
    game_durations_won.rename(columns={'game_difficulty': 'Level Difficulty'}, inplace=True)
    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    ax.yaxis.grid(True)
    colors = [paper_plot_utils.C0, '#F7F7F7', paper_plot_utils.C1]
    sns.histplot(data=game_durations_won, ax=ax, multiple='stack', x='time', kde=False, binwidth=3, hue='Level Difficulty', element='bars',
                 legend=True, palette=colors, zorder=10)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Count')
    plt.savefig('imgs/performance/trial_times_hist.png')
    plt.savefig('../paper/trial_times_hist.svg', format="svg")
    plt.show()


def add_estimated_level_scores(df):
    return add_level_score_estimations_to_df(df)


def plot_mean_score_per_level():
    df = pd.read_csv('../data/level_scores.csv').drop_duplicates()

    print('Mean Score Per Level: ', df['level_score'].mean())
    print('Var Score Per Level: ', df['level_score'].var())
    print('Median Score Per Level: ', df['level_score'].median())

    fig, ax = plt.subplots(figsize=paper_plot_utils.figsize)
    order_df = df.groupby(['subject_id'])['level_score'].mean().reset_index().sort_values('level_score')
    sns.pointplot(data=df, ax=ax, x='subject_id', y='level_score', palette=[paper_plot_utils.C0], markers='.', capsize=.6, errwidth=0.8,
                  join=False,
                  errorbar='se', order=order_df['subject_id'])
    sns.despine(bottom=True, left=True)
    ax.set_ylabel('Mean score per level')
    ax.set_xlabel('Subjects')

    xlabels = [subject2letter(subj_id.get_text()) for subj_id in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)

    plt.tight_layout()
    plt.savefig('imgs/performance/score_per_level.png')
    plt.savefig('../paper/score_per_level.svg', format="svg")
    plt.show()


def save_level_scores():
    df = read_data()
    df = add_level_score_estimations_to_df(df)
    scores_per_level = df[['subject_id', 'game_difficulty', 'world_number', 'level_score', 'score']]
    scores_per_level.to_csv('level_scores.csv', index=False)
    print('Saved Level Scores')


def anova_mean_level_score():
    print('H0: Means for score per level are equal | H1: Means are different for each subject')

    df = pd.read_csv('../data/level_scores.csv').drop_duplicates()

    scores = df.groupby('subject_id')['level_score'].apply(list).tolist()

    anova_result = scipy.stats.f_oneway(*scores)
    print(anova_result)
    print('dof=', len(scores) - 1)  # TODO correct?


def ttest_mean_time_easy_normal():
    df = pd.read_csv('../data/game_durations.csv')

    # get weighted fixation distances
    df_easy = df[df['game_difficulty'] == 'easy']
    df_normal = df[df['game_difficulty'] == 'normal']

    times_easy = df_easy['time']
    times_normal = df_normal['time']

    print('H0: Same Means | H1: Mean Game Durations for easy levels less than for medium levels')

    # perform (Welch's) t-test
    ttest_result = scipy.stats.ttest_ind(times_easy, times_normal, alternative='less')  # use equal_var=False bc of different sample sizes
    print(ttest_result)
    print('dof=', len(times_easy) - 1 + len(times_normal) - 1)


def ttest_mean_time_normal_hard():
    df = pd.read_csv('../data/game_durations.csv')

    # get weighted fixation distances
    df_normal = df[df['game_difficulty'] == 'normal']
    df_hard = df[df['game_difficulty'] == 'hard']

    times_normal = df_normal['time']
    times_hard = df_hard['time']

    print('H0: Same Means | H1: Mean Game Durations for medium levels less than for hard levels')

    # perform (Welch's) t-test
    ttest_result = scipy.stats.ttest_ind(times_normal, times_hard, alternative='less')  # use equal_var=False bc of different sample sizes
    print(ttest_result)
    print('dof=', len(times_normal) - 1 + len(times_hard) - 1)


if __name__ == '__main__':
    print_average_game_endings()
    # histogram_over_avg_trial_times()
    # plot_mean_score_per_level()

# plot_performance_per_difficulty()
