import config


def get_times_from_samples(samples):
    return samples.T[0]


def get_gaze_coords_from_samples(samples):
    xy_coords = samples.T[1:3]
    return xy_coords[0], xy_coords[1]


def get_pupil_size_from_samples(samples):
    return samples.T[3]


def plot_gaze(ax, samples):

    coords = get_gaze_coords_from_samples(samples)

    ax.plot(*coords, color='black')
    ax.scatter(*coords, color='black')

    ax.set_xlim(0, config.DISPLAY_WIDTH_PX)
    ax.set_ylim(0, config.DISPLAY_HEIGHT_PX)


def plot_pupil_size_over_time(ax, samples):
    times = get_times_from_samples(samples)
    pupil_size = get_pupil_size_from_samples(samples)

    ax.plot(times, pupil_size, color='black')
