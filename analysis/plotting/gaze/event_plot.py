def plot_blinks(ax, times, blink_start, blink_end):

    for i, b_start in enumerate(blink_start):
        b_current_plot = b_start
        # while b_current_plot <= b_start + blink_end[i]:
        #     ax.axvline(b_current_plot, color='black')
        #     b_current_plot += 1
        ax.hlines(y=1, xmin=b_start, xmax=b_start + blink_end[i], linewidth=2, color='r')

    ax.set_xlim((0, max(times)))
    ax.set_xlabel('time')
    ax.set_ylabel('blinks')

