from pygaze import libtime, libscreen

from pylink import *
from pygaze.eyetracker import EyeTracker

# eyetracker events
EVENTS = [STARTFIX, STARTBLINK, STARTSACC, ENDFIX, ENDSACC, ENDBLINK]


class MyEyeTracker:

    def __init__(self, edf_path='./edf_files/test.edf'):
        self.disp = self.eye_tracker_disp = libscreen.Display()
        print('Creating eyetracker instance...')
        self.tracker = EyeTracker(self.disp, data_file=edf_path)
        print('EyeTracker created.')
        self.t_start = None  # start time of tracking
        self.eyetracker_events = None  # accumulator for events during recording

    def __del__(self):
        self.close()
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
        """ Gets the latest sample and returns it as (time, 3gaze, pupil_size). """
        sample_time = self.get_time()
        gaze = self.tracker.sample()
        pupil_size = self.tracker.pupil_size()

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
        """ Extracts events from the eye tracker and saves them in a list. """

        for i in range(3):
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
