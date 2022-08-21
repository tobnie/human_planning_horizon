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
            "target_position": game.world.get_target_position(),
            "n_lanes": config.N_LANES,
            "n_water_lanes": config.N_WATER_LANES,
            "n_street_lanes": config.N_STREET_LANES,
            "field_width": config.FIELD_WIDTH,
            "field_height": config.FIELD_HEIGHT,
            "player_width": config.PLAYER_WIDTH,
            "player_height": config.PLAYER_HEIGHT,
            "display_x": config.DISPLAY_WIDTH_PX,
            "display_y": config.DISPLAY_HEIGHT_PX,
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
        self.eyetracker_samples.append((time, gaze_x, gaze_y, pupil_size))

    def log_eyetracker_events(self, eyetracker_events):
        """ Logs the eyetracker events. """
        self.eyetracker_events = eyetracker_events

    def save_data(self, training=False):
        """ Saves the data to files. The world properties are saved in a .csv-file,
        while the world states and player actions are saved in .npz-files """

        # save world properties
        self._save_properties_as_csv(training=training)

        # save world states as .npz-file
        self._save_world_states_as_npz(training=training)

        # save actions as .npz-file
        self._save_actions_as_npz(training=training)

        if self.eyetracker_samples:
            # save eyetracking data as .npz file
            self._save_eyetracker_samples_as_npz(training=training)
        else:
            warnings.warn("No eye tracking Samples to save.", RuntimeWarning)

        if self.eyetracker_events:
            # save eyetracking events as .npz file
            self._save_eyetracker_events_as_npz(training=training)
        else:
            warnings.warn("No eye tracking Events to save.", RuntimeWarning)

    def _save_properties_as_csv(self, training=False):
        """ Saves the game properties as a .csv-file. """
        log_directory = self.log_directory if not training else self.training_log_directory
        with open(log_directory + 'world_properties.csv', 'w') as f:
            for key in self.game_properties.keys():
                f.write("%s, %s\n" % (key, self.game_properties[key]))

    def _save_actions_as_npz(self, training=False):
        """ Saves the player actions as a .npz-file. """

        if not self.player_actions:
            warnings.warn("No player actions to save.", RuntimeWarning)
            return

        times, actions = zip(*self.player_actions)

        # concatenate time and state
        actions_with_time = np.vstack((times, actions)).T

        # save as .npz-file
        log_directory = self.log_directory if not training else self.training_log_directory
        np.savez_compressed(log_directory + 'actions.npz', actions_with_time)

    def _save_world_states_as_npz(self, training=False):
        """ Saves the world states as a .npz-file. """
        log_directory = self.log_directory if not training else self.training_log_directory

        # state_directory = log_directory + 'states/'
        # if not os.path.exists(state_directory):
        #     os.makedirs(state_directory)

        # for time, state in self.world_states:
        #   np.savez_compressed(state_directory + f'state_{time}.npz', state)

        times, states = list(zip(*self.world_states))
        times = np.array(times)
        states = np.array(states)

        npz_dict = {'times': times, 'states': states}
        np.savez_compressed(log_directory + f'world_states.npz', **npz_dict)

    def _save_eyetracker_samples_as_npz(self, training=False):
        """ Saves the eyetracker data as a .npz-file. """

        times, gaze_x, gaze_y, pupil_size = zip(*self.eyetracker_samples)

        # concatenate time and state
        gaze_with_time = np.vstack((times, gaze_x, gaze_y, pupil_size)).T

        # save as .npz-file
        log_directory = self.log_directory if not training else self.training_log_directory
        np.savez_compressed(log_directory + 'eyetracker_samples.npz', gaze_with_time)

    def _save_eyetracker_events_as_npz(self, training=False):
        """ Saves the eyetracker events as a .npz-file. """

        times, events = zip(*self.eyetracker_events)

        # concatenate time and state
        events_with_time = np.vstack((times, events)).T

        # save as .npz-file
        log_directory = self.log_directory if not training else self.training_log_directory
        np.savez_compressed(log_directory + 'eyetracker_events.npz', events_with_time)

    def _set_log_directory(self, log_directory, subject_id, difficulty, world_name):
        """ Sets the log directory as 'log_directory/subject_id/' and creates it if not existent. """
        self.training_log_directory = log_directory + subject_id + '/training/' + difficulty.value + '/' + world_name + '/'
        self.log_directory = log_directory + subject_id + '/' + difficulty.value + '/' + world_name + '/'

        # check if paths exist
        if not os.path.exists(self.log_directory):
            # Create a new directory if it does not exist
            os.makedirs(self.log_directory)

        if not os.path.exists(self.training_log_directory):
            # Create a new directory if it does not exist
            os.makedirs(self.training_log_directory)
