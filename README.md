# human_planning_horizon

For saving the data in central files to load as a single dataframe from, run ``save_compressed_data.py`` in ``/data`` first (run from inside ``/data`` please). This will generate compressed files for each subject in ``data/compressed_data/``. This might take a while.

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
| player_x_field  | x coordinate of player in field coordinates                                                  |
| player_y_field  | y coordinate of player in field coordinates                                                  |
| action          | action int                                                                                   |
| game_status     | whether the game the current row belongs to was 'won', 'lost' or 'timed_out'                 |
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

To load a dataframe containing all score information, call ``read_score_data()`` in ``analysis/score_utils.py``. The resulting dataframe is structured like this:

| entry name      | description                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------|
| subject_id      | id of the subject                                                                            |
| after_levels    | number of levels after which this score was achieved                                         |
| score           | score reached after a certain number of levels (cf. 'after_levels')                          |


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

# Processing of Eye Data

Eye Data can be processed via ``analysis/plotting/gaze/events/event_detection.py`` for saccades, blinks and fixations.

## Plots

Plots are located in ``analysis/imgs/``. **Since I have made a lot of changes to the way data is saved or loaded, not every method for creating plots may work as expected and might need some fixes or tweaks**.
