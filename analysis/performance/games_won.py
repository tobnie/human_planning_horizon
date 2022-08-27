import config
from analysis.data_utils import read_data

WIN_THRESHOLD_Y = (config.N_LANES - 2) * config.FIELD_HEIGHT
TIME_OUT_THRESHOLD = config.LEVEL_TIME - 100


def get_last_time_steps_of_games(df):
    # get indices of last dataframe row of each game
    last_time_steps = df.groupby(['subject_id', 'game_difficulty', 'world_number']).tail(1)
    return last_time_steps


def get_won_games(df=None):
    if df is None:
        df = read_data()
    last_game_steps_df = get_last_time_steps_of_games(df)
    won_games = last_game_steps_df.loc[last_game_steps_df['player_y'] >= WIN_THRESHOLD_Y]
    return won_games


def get_lost_games(df):
    last_game_steps_df = get_last_time_steps_of_games(df)
    non_won_games = last_game_steps_df.loc[last_game_steps_df['player_y'] < WIN_THRESHOLD_Y]
    lost_games = non_won_games.loc[non_won_games['time'] < TIME_OUT_THRESHOLD]
    return lost_games


def get_timed_out_games(df):
    last_game_steps_df = get_last_time_steps_of_games(df)
    non_won_games = last_game_steps_df.loc[last_game_steps_df['player_y'] < WIN_THRESHOLD_Y]
    timed_out_games = non_won_games.loc[df['time'] >= TIME_OUT_THRESHOLD]
    return timed_out_games


# df = read_data()
# # group by game_difficulty:
# df = df.groupby(['game_difficulty']).apply(get_won_games)
# counts = df['game_difficulty'].value_counts()
# print(counts)
