from analysis.data_utils import read_data


def print_avg_time_on_middle_lane():
    df = read_data()
    middle_lane_times = df.groupby(['subject_id', 'game_difficulty', 'world_number']).apply(calc_portion_on_middle_lane)
    print(middle_lane_times)
    print('Average: ', middle_lane_times.mean())
    print('Var:', middle_lane_times.var())
    print('Median:', middle_lane_times.median())


def calc_portion_on_middle_lane(game_df):
    samples = game_df.shape[0]
    middle_lane_samples = game_df[game_df['region'] == 'middle'].shape[0]
    return middle_lane_samples / samples


if __name__ == '__main__':
    print_avg_time_on_middle_lane()
