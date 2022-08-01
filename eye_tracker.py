from pygaze import libtime

import config
from pylink import *
from pygaze.eyetracker import EyeTracker

# eyetracker events
EVENTS = [STARTFIX, STARTBLINK, STARTSACC, ENDFIX, ENDSACC, ENDBLINK]


class MyEyeTracker:

    # TODO only track dominant eye or in binocular mode or ...?

    def __init__(self, disp, edf_path='./edf_files/test.edf'):
        self.disp = disp  # TODO maybe this can even be declared and initialized here instead of an actual parameter
        print('Creating eyetracker instance...')
        self.tracker = EyeTracker(self.disp, data_file=edf_path,
                                  resolution=(config.DISPLAY_WIDTH_PX, config.DISPLAY_HEIGHT_PX))
        print('EyeTracker created.')
        self.t_start = None  # start time of tracking
        self.eyetracker_events = None  # accumulator for events during recording

    def __del__(self):
        # self.close()
        print('EyeTracker instance deleted.')

    def calibrate(self):
        """ Starts the calibration menu of the eye tracker. """
        return self.tracker.calibrate()

    def start_recording(self):
        """ Starts recording of the eye tracker. """

        if not self.connected():
            raise RuntimeError("EyeTracker is not connected.")

        self.eyetracker_events = []
        libtime.expstart()
        self.t_start = getEYELINK().getCurrentTime()
        # TODO what exactly happens at drift correction and do we need that?
        # self.tracker.drift_correction()
        self.tracker.start_recording()

    def get_time(self):
        """ Returns the current time relative to the start of the experiment. """
        return getEYELINK().getCurrentTime() - self.t_start

    def show_status_message(self, msg):
        """ Shows a status message on the experimenter's screen. """
        self.tracker.status_msg(msg)

    def connected(self):
        """ Returns True, if the eye tracker is currently connected. False, otherwise."""
        return self.tracker.connected()

    def get_sample(self):
        """ Gets the latest sample and returns it as (time, gaze, pupil_size). """
        sample_time = self.get_time()
        gaze = self.tracker.sample()
        pupil_size = self.tracker.pupil_size()

        print("----------------------\n"
              f"Time: {sample_time}\n"
              f"Gaze: {gaze}\n"
              f"Pupil Size: {pupil_size}\n"
              "----------------------")

        return sample_time, gaze, pupil_size

    def log_var(self, varname, varvalue):
        """ Logs a variable to the eye tracker. """
        self.tracker.log_var(varname, varvalue)

    def log(self, msg):
        """ Logs a message."""
        self.tracker.log(msg)

    def status_msg(self, msg):
        """ Shows a status message on the experimenter's screen. """
        self.tracker.status_msg(msg)

    def stop_recording(self):
        """ Stops recording of the eye tracker and returns all recorded eye tracking events. """
        self.tracker.stop_recording()
        return self.eyetracker_events

    def close(self):
        """ Closes the eye tracker connection. """
        self.tracker.close()

    def extract_events(self):
        """ Extracts events from the eye tracker and returns them as a DataFrame. """

        for i in range(3):  # TODO why for i in range(3)? --> because it only looks at the next 3 events?
            tc, d = self.get_event()
            if tc is None and d is None:
                pass
            else:
                self.eyetracker_events.append([tc, d])

    def get_event(self):
        """ Returns the next event from the eye tracker. """
        d = getEYELINK().getNextData()
        if d in EVENTS:
            float_data = getEYELINK().getFloatData()
            # corresponding clock_time
            tc = float_data.getTime() - self.t_start
            return tc, d
        return None, None

