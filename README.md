# human_planning_horizon

For saving the data in central files to load as a single dataframe from, run ``save_compressed_data.py`` in ``/data`` first (run from inside ``/data`` please). This will generate compressed files for each subject in ``data/compressed_data/``.

The data can be loaded via ``read_subject_data(subject_id)`` or ``read_data()`` for all subjects in ``analysis/data_utils.py`` (also run within subdirectory please).

## Content of Dataframes

| entry name    | description                                                                           |
|-----------------|----------------------------------------------------------------------------------------------|
| game_difficulty | 'easy', 'medium' or 'hard'                                                                   |
| world_number    | in [0, 19]. there are 20 worlds per difficulty                                               |
| target_position | int. Describes the field in x-coords in the last row of the game, that the player must reach |
| time            | time stamp of the row data                                                                   |
| gaze_x          | x coordinate of gaze                                                                         |
| gaze_y          | y coordinate of gaze                                                                         |
| pupil_size      | pupil size in mm^2                                                                           |
| player_x        | x coordinate of player                                                                       |
| player_y        | y coordinate of player                                                                       |
| action          | action int                                                                                   |
| state           | array representing the state. see below for explanation                                      |

## Representation of State:

| state[:, 0] | state[:, 1] | state[:, 2]| state[:,3] |
|-----------------|----------------------|----------------------------------|--------------------------------------|
| object_type (int) | x (float) | y (float) | width (int) |

where the object type is encoded as

| int | type    |
|-----|---------|
| 0   | player  |
| 1   | vehicle |
| 2   | lilypad |

# Score Data

Score data is located in  ``data/scores``. The score is saved after the first and after each five levels. Please keep in mind that the files also contain dummy data that is shown during the experiment, besides the actual data.
There does not exist a score for every subject since there were errors in the calculcation of the score in the first few experiments.

# SoSci Survey Data

The SoSci Survey Data is located in ``data/sosci_data.csv``. Column 'SD_05' corresponds to how often they play video games.

| int | corresponding answer   |
|-----|---------|
| 1   | never |
| 2   | rarely |
| 3   | sometimes |
| 4   | often |
| 5   | daily |
