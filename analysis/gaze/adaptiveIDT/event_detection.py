from analysis.data_utils import read_subject_data
from analysis.gaze.adaptiveIDT.data_filtering import savitzky_golay_filter


def event_detection_for_trial(trial_df):

    x, y = trial_df['gaze_x'].to_numpy(), trial_df['gaze_y'].to_numpy()

    filtered_x, filtered_y = savitzky_golay_filter(x, y)

    pass


if __name__ == '__main__':
    df = read_subject_data('ED06RA')
    trials = df.groupby(['trial'])
    detected_events = trials.apply(event_detection_for_trial)
    print(detected_events)