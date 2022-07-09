import os

import numpy as np

import config


class Logger:
    """ Logs the game properties and world states of the current game."""

    def __init__(self, game, subject_id, world_name, difficulty, time_limit, log_directory=config.LEVEL_DATA_DIR):
        # set log directory
        self._set_log_directory(log_directory, subject_id, world_name)
        self.game = game
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
        self.eyetracker_data = []  # list of (time, gaze_x, gaze_y) # TODO extend

    def log_state(self,):
        """ Logs the current time and world state """
        self.world_states.append((self.game.game_time, self.world.get_world_state().asarray()))

    def log_action(self):
        """ Logs the current time and action """
        self.player_actions.append((self.game.game_time, self.world.player.get_action().value))

    def log_eyetracker_data(self, eyetracker_data):
        """ Logs the current time and given eyetracker data. """
        gaze_x = np.random.rand()
        gaze_y = np.random.rand()
        self.eyetracker_data.append((self.game.game_time, gaze_x, gaze_y)) # TODO REMOVE DUMMY DATA

    def save_data(self):
        """ Saves the data to files. The world properties are saved in a .csv-file,
        while the world states and player actions are saved in .npz-files """

        # save world properties
        self._save_properties_as_csv()

        # save world states as .npz-file
        self._save_world_states_as_npz()

        # save actions as .npz-file
        self._save_actions_as_npz()

        # save eyetracking data as .npz file
        self._save_eyetracker_data_as_npz()

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

        # TODO remove print
        print("Full array Actions:\n", actions_with_time)

        # save as .npz-file
        np.savez_compressed(self.log_directory + 'actions.npz', actions_with_time)

    def _save_world_states_as_npz(self):
        """ Saves the world states as a .npz-file. """

        times, states = zip(*self.world_states)
        states = np.asarray(states)

        # concatenate time and state
        times, states = np.asarray(times).reshape(-1, 1), states.reshape(states.shape[0], states.shape[2])
        states_with_time = np.hstack((times, states))

        # TODO remove print
        print("Full array state:\n", states_with_time)

        # save as .npz-file
        np.savez_compressed(self.log_directory + 'world_states.npz', states_with_time)

    def _save_eyetracker_data_as_npz(self):
        """ Saves the eyetracker data as a .npz-file. """

        times, gaze_x, gaze_y = zip(*self.eyetracker_data)

        # concatenate time and state
        gaze_with_time = np.vstack((times, gaze_x, gaze_y)).T

        # TODO remove print
        print("Full array EyeTracking:\n", gaze_with_time)

        # save as .npz-file
        np.savez_compressed(self.log_directory + 'eyetracker_data.npz', gaze_with_time)

    def _set_log_directory(self, log_directory, subject_id, world_name):
        """ Sets the log directory as 'log_directory/subject_id/' and creates it if not existent. """
        self.log_directory = log_directory + subject_id + '/' + world_name + '/'

        # check if path exists
        if not os.path.exists(self.log_directory):
            # Create a new directory if it does not exist
            os.makedirs(self.log_directory)
