from analysis.data_utils import read_data, get_last_time_steps_of_games, WIN_THRESHOLD_Y, TIME_OUT_THRESHOLD


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