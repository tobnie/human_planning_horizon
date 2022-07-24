import os
import warnings

import numpy as np

import config


class Logger:
    """ Logs the game properties and world states of the current game."""

    def __init__(self, game, subject_id, world_name, difficulty, time_limit, log_directory=config.LEVEL_DATA_DIR):
        # set log directory
        self._set_log_directory(log_directory, subject_id, difficulty, world_name)
        self.world = game.world

        # logging_game variables
        self.game_properties = {
            "subject_id": subject_id,
            "world_name": world_name,
            "difficulty": difficulty.value,
            "time_limit": time_limit,
        }  # general world properties
        self.world_states = []  # list of world states as (time, world_state)
        self.player_actions = []  # list of (time, action)
        self.eyetracker_samples = []  # list of (time, gaze_x, gaze_y, pupil_size)
        self.eyetracker_events = []  # list of (time, event)

    def log_state(self, time):
        """ Logs the current time and world state """
        self.world_states.append((time, self.world.get_world_state().asarray()))

    def log_action(self, time):
        """ Logs the current time and action """
        self.player_actions.append((time, self.world.player.get_action().value))

    def log_eyetracker_samples(self, time, sample):
        """ Logs the current time and given eyetracker data. """
        (gaze_x, gaze_y), pupil_size = sample
        self.eyetracker_samples.append((time, gaze_x, gaze_y, pupil_size))  # TODO REMOVE DUMMY DATA

    def log_eyetracker_events(self, eyetracker_events):
        """ Logs the eyetracker events. """
        self.eyetracker_events = eyetracker_events

    def save_data(self):
        """ Saves the data to files. The world properties are saved in a .csv-file,
        while the world states and player actions are saved in .npz-files """

        # save world properties
        self._save_properties_as_csv()

        # save world states as .npz-file
        self._save_world_states_as_npz()

        # save actions as .npz-file
        self._save_actions_as_npz()

        if self.eyetracker_samples:
            # save eyetracking data as .npz file
            self._save_eyetracker_samples_as_npz()
        else:
            warnings.warn("No eye tracking Samples to save.", RuntimeWarning)

        if self.eyetracker_events:
            # save eyetracking events as .npz file
            self._save_eyetracker_events_as_npz()
        else:
            warnings.warn("No eye tracking Events to save.", RuntimeWarning)

    def _save_properties_as_csv(self):
        """ Saves the game properties as a .csv-file. """
        with open(self.log_directory + 'world_properties.csv', 'w') as f:
            for key in self.game_properties.keys():
                f.write("%s, %s\n" % (key, self.game_properties[key]))

    def _save_actions_as_npz(self):
        """ Saves the player actions as a .npz-file. """

        times, actions = zip(*self.player_actions)

        # concatenate time and state
        actions_with_time = np.vstack((times, actions)).T

        # save as .npz-file
        np.savez_compressed(self.log_directory + 'actions.npz', actions_with_time)

    def _save_world_states_as_npz(self):
        """ Saves the world states as a .npz-file. """
        for time, state in self.world_states:
            np.savez_compressed(self.log_directory + f'states/state_{time}.npz', state)

    def _save_eyetracker_samples_as_npz(self):
        """ Saves the eyetracker data as a .npz-file. """

        times, gaze_x, gaze_y, pupil_size = zip(*self.eyetracker_samples)

        # concatenate time and state
        gaze_with_time = np.vstack((times, gaze_x, gaze_y, pupil_size)).T

        # TODO remove print
        # print("Full array EyeTracking:\n", gaze_with_time)

        # save as .npz-file
        np.savez_compressed(self.log_directory + 'eyetracker_samples.npz', gaze_with_time)

    def _save_eyetracker_events_as_npz(self):
        """ Saves the eyetracker events as a .npz-file. """

        times, events = zip(*self.eyetracker_events)

        # concatenate time and state
        events_with_time = np.vstack((times, events)).T

        # save as .npz-file
        np.savez_compressed(self.log_directory + 'eyetracker_events.npz', events_with_time)

    def _set_log_directory(self, log_directory, subject_id, difficulty, world_name):
        """ Sets the log directory as 'log_directory/subject_id/' and creates it if not existent. """
        self.log_directory = log_directory + subject_id + '/' + difficulty.value + '/' + world_name + '/'

        # check if path exists
        if not os.path.exists(self.log_directory):
            # Create a new directory if it does not exist
            os.makedirs(self.log_directory)


# TODO translation from event codes to actual event in analysis (or even change here, dont know yet)
from pylink import *


def get_event_string(event):
    if event == STARTFIX:
        return ["fixation", "start"]
    if event == STARTSACC:
        return ["saccade", "start"]
    if event == STARTBLINK:
        return ["blink", "start"]
    if event == ENDFIX:
        return ["fixation", "end"]
    if event == ENDSACC:
        return ["saccade", "end"]
    if event == ENDBLINK:
        return ["blink", "end"]
