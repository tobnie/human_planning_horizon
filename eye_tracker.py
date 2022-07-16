import pylink
from pygaze.eyetracker import EyeTracker


class MyEyeTracker:

    def __init__(self, disp):
        self.disp = disp
        self.tracker = EyeTracker(disp, trackertype="dummy")

    def calibrate(self):
        """ Starts the calibration menu of the eye tracker. """
        self.tracker.calibrate()

    def start_recording(self):
        """ Starts recording of the eye tracker. """
        # TODO what exactly happens at drift correction and do we need that?
        self.tracker.drift_correction()
        self.tracker.start_recording()

    def show_status_message(self, msg):
        """ Shows a status message on the experimenter's screen. """
        self.tracker.status_msg(msg)

    def get_sample(self):
        """ Returns the latest sample from the eye tracker as (x, y)-tuple. """
        return self.tracker.sample()

    def _get_samples(self):
        # wait for eye movement
        t1, startpos = self.tracker.wait_for_saccade_start()
        endtime, startpos, endpos = self.tracker.wait_for_saccade_end()

    def stop_recording(self):
        """ Stops recording of the eye tracker. """
        self.tracker.stop_recording()


# class MyEyeTrackerPyLink:
#
#     def __init__(self, ip='100.1.1.1'):
#         self.tracker = pylink.EyeLink(ip)
#         self.t_start = None
#
#         # TODO does file need to exist? new file for every trial? how to synchronize with game time then?
#         self.tracker.openDataFile('test.edf')
#
#     def configure(self, sample_rate, screen_width, screen_height):
#         # put into offline mode before sending commands
#         self.tracker.setOfflineMode()
#         # Set sample rate
#         self.tracker.sendCommand(f"sample_rate {sample_rate}")
#         # Send screen resolution to the tracker
#         self.tracker.sendCommand(f'screen_pixel_coords = 0 0 {screen_width - 1} {screen_height - 1}')
#         # Set the calibration type to 9-point (HV9)
#         self.tracker.sendCommand("calibration_type = HV9")
#
#     def calibrate(self):
#         # Step 4: open a calibration window
#         pylink.openGraphics()  # TODO ?
#         self.tracker.doTrackerSetup()
#
#     def drift_correction(self):
#         self.tracker.do
#
#     def log_message(self, msg):
#         self.tracker.sendMessage(msg)
#
#     def start(self):
#         # start recording
#         self.tracker.startRecording(1, 1, 1, 1)
#         self.t_start = self.tracker.trackerTime()
#
#     def get_sample(self):
#         # TODO not only get samples but also events
#         # get the latest sample
#         # Current tracker time
#
#         smp_time = -1
#         while True:
#             # Poll the latest samples
#             smp = self.tracker.getNewestSample()
#             if smp is not None:
#                 # Grab gaze, HREF, raw, & pupil size data
#                 # TODO binocular mode
#                 if smp.isRightSample():
#                     gaze = smp.getRightEye().getGaze()
#                     href = smp.getRightEye().getHREF()
#                     raw = smp.getRightEye().getRawPupil()
#                     pupil = smp.getRightEye().getPupilSize()
#                 elif smp.isLeftSample():
#                     gaze = smp.getLeftEye().getGaze()
#                     href = smp.getLeftEye().getHREF()
#                     raw = smp.getLeftEye().getRawPupil()
#                     pupil = smp.getLeftEye().getPupilSize()
#
#                 timestamp = smp.getTime()
#
#                 # Save gaze, HREF, raw, & pupil data to the plain text
#                 # file, if the sample is new
#                 if timestamp > smp_time:
#                     smp_data = map(str, [timestamp, gaze, href, raw, pupil])
#                     text_file.write('\t'.join(smp_data) + '\n')
#                     smp_time = timestamp
#
#     def stop(self):
#         # stop recording
#         self.tracker.stopRecording()
#         self.tracker.closeDataFile()
#         self.tracker.receiveDataFile('test.edf', 'test.edf')  # TODO  ?
#
#     def __del__(self):
#         # close the link to the tracker, then close the window
#         self.tracker.close()
#         pylink.closeGraphics()  # TODO ?
