import config


def calc_score(game_df):
    diff2multiplier = {'easy': config.EASY_MULTIPLIER, 'normal': config.NORMAL_MULTIPLIER, 'hard': config.HARD_MULTIPLIER}
    difficulty_multiplier = diff2multiplier[game_df['game_difficulty'].values[0]]

    game_status = game_df['game_status'].values[0]

    # dead
    if game_status == 'lost':
        return config.DEATH_PENALTY
    # won
    elif game_status == 'won':
        lane_score = 14 * config.VISITED_LANE_BONUS
        win_bonus = config.WIN_BONUS
        remaining_time = (config.LEVEL_TIME - game_df['time'].max()) / 1_000
        return (lane_score + win_bonus + remaining_time) * difficulty_multiplier
    # timed out
    elif game_status == 'timed_out':
        # highest lane times 5
        score = game_df['player_y_field'].max() * config.VISITED_LANE_BONUS
        return score * difficulty_multiplier
    else:
        raise NotImplementedError("Case is not covered by method")


def add_level_score_estimations_to_df(df):
    level_scores = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_score).reset_index()
    level_scores.columns = ['subject_id', 'game_difficulty', 'world_number', 'level_score']
    df = df.merge(level_scores, on=['subject_id', 'game_difficulty', 'world_number'], how='left')
    return df


def add_estimated_scores_when_missing(df):
    nan_scores = df[df['score'].isna()]
    scores = nan_scores.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_score).reset_index()
    scores.columns = ['subject_id', 'game_difficulty', 'world_number', 'score']
    scores = scores[['subject_id', 'score']].groupby(['subject_id']).sum().reset_index()

    # remove incomplete subject scores
    scores = scores[(scores['subject_id'] != 'ZI01SU') & (scores['subject_id'] != 'NI07LU')]

    for subject in scores['subject_id'].unique():
        mask = df['subject_id'] == subject
        score = scores[scores['subject_id'] == subject]['score']
        score_value = score.iloc[0]
        df.loc[mask, 'score'] = score_value
    return df
