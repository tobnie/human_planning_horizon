import pylink
from pygaze.eyetracker import EyeTracker


class MyEyeTracker:

    # TODO only track dominant eye or in binocular mode or ...?

    def __init__(self, disp):
        self.disp = disp
        print('Creating eyetracker instance...')
        self.tracker = EyeTracker(self.disp, trackertype="dumbdummy")
        print('EyeTracker created.')

    def reset_and_recalibrate(self):
        """ Closes the eye tracker connection and establishes a new one. This is useful since data is only stored when calling close()."""
        # TODO is the last sentence in the doc correct?
        self.tracker.close()
        self.tracker = EyeTracker(self.disp, trackertype="dumbdummy")
        self.calibrate()    # TODO do I need to calibrate again after establishing a new connection?
        # TODO drift correction?

    def calibrate(self):
        """ Starts the calibration menu of the eye tracker. """
        self.tracker.calibrate()

    def start_recording(self):
        """ Starts recording of the eye tracker. """

        if not self.connected():
            raise RuntimeError("EyeTracker is not connected.")

        # TODO what exactly happens at drift correction and do we need that?
        self.tracker.drift_correction()
        self.tracker.start_recording()

    def show_status_message(self, msg):
        """ Shows a status message on the experimenter's screen. """
        self.tracker.status_msg(msg)

    def connected(self):
        """ Returns True, if the eye tracker is currently connected. False, otherwise."""
        return self.tracker.connected()

    def get_sample(self):
        """ Returns the latest sample from the eye tracker as (x, y)-tuple. """
        # TODO do we need that if we have a separate logging file for the eyetracking data including all events etc.?
        return self.tracker.sample()

    def pupil_size(self):
        """ Returns pupilsize for the currently tracked eye"""
        return self.tracker.pupil_size()

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
        """ Stops recording of the eye tracker. """
        self.tracker.stop_recording()
