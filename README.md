# Do humans adapt their Planning Horizon? - An Analysis of Sequential Decision-Making in the videogame Frogger 

## Abstract

Humans can employ sophisticated strategies to plan their next actions while only having limited cognitive
capacity. Most of the studies investigating human behavior focus on minimal and rather abstract tasks. We
provide an environment inspired by the video game Frogger, especially designed for studying how far humans
plan ahead. This is described by the planning horizon, a not directly measurable quantity representing how
far ahead people can consider the consequences of their actions. We treat the subjects’ eye movements as
externalization of their internal planning horizon and can thereby infer its development over time.

We found that people can dynamically adapt their planning horizon when switching between tasks. While
subjects employed bigger planning horizons, we could not measure any physiological indicators of stronger
cognitive engagement. Subjects using larger planning horizons were able to score higher in the game. We
designed neural networks for predicting the subject’s planning horizon in different situations, providing
further insight on the key features needed for accurate predictions of the planning horizon. In general, models
trained only on subject-specific data achieved higher accuracy than models trained on data collected from all
subjects.

## Usage

For saving the data in central files to load as a single dataframe from, run ``run_preprocessing.py`` in ``/data`` first (run from inside ``/data`` please). This will generate compressed files for each subject in ``data/compressed_data/``. This might take a while.

The data can be loaded via ``read_subject_data(subject_id)`` or ``read_data()`` for all subjects in ``analysis/data_utils.py`` (also run within subdirectory please).

### Content of Dataframes

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

### Representation of State:

| state[:, 0] | state[:, 1] | state[:, 2]| state[:,3] |
|-----------------|----------------------|----------------------------------|--------------------------------------|
| object_type (int) | x (float) | y (float) | width (int) |

where the object type is encoded as

| int | type    |
|-----|---------|
| 0   | player  |
| 1   | vehicle |
| 2   | lilypad |

### Score Data

To load a dataframe containing all score information, call ``read_score_data()`` in ``analysis/score_utils.py``. The resulting dataframe is structured like this:

| entry name      | description                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------|
| subject_id      | id of the subject                                                                            |
| after_levels    | number of levels after which this score was achieved                                         |
| score           | score reached after a certain number of levels (cf. 'after_levels')                          |


Score data is located in  ``data/scores``. The score is saved after the first and after each five levels. Please keep in mind that the files also contain dummy data that is shown during the experiment, besides the actual data.
There does not exist a score for every subject since there were errors in the calculcation of the score in the first few experiments.

### Trial Order

The trial order can be loaded as a dataframe with ``load_trial_orders()`` in ``analysis/trial_order_utils.py``. It returns a dataframe of the following structure:

| entry name      | description                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------|
| subject_id      | id of the subject                                                                            |
| trial           | number of trial (maximum is 60)                                                              |    
| game_difficulty      | difficulty of the trial                                                                      |
| world_number    | world number of the trial                                                                    |

The trial order for each subject is saved in ``data/level_data/<subject_id>/trial_order.csv``. The csv has the following structure (separated by ';'):

| csv[:, 0] | csv[:, 1] | csv[:, 2] |
| --- | --- | --- |
| trial_nr | difficulty | world_name |

The ``world_name`` is a string of form 'world_<world_nr>'.

**Not every subject has a ``trial_order.csv`` since this was accidentally not saved for the first few experiments.

## SoSci Survey Data

The SoSci Survey Data is located in ``data/sosci_data.csv``. Column 'SD_05' corresponds to how often they play video games.

| int | corresponding answer   |
|-----|---------|
| 1   | never |
| 2   | rarely |
| 3   | sometimes |
| 4   | often |
| 5   | daily |

## Processing of Eye Data

Eye Data can be processed via ``analysis/plotting/gaze/events/event_detection.py`` for saccades, blinks and fixations.

### Plots

Plots are located in ``analysis/imgs/``. **Since I have made a lot of changes to the way data is saved or loaded, not every method for creating plots may work as expected and might need some fixes or tweaks**.

## Remarks to Subject Data

The data of subject ``NI07LU`` is not complete, since the experiment crashed.

Some data of subject ``ZI01SU`` is missing because of errors in the code during the experiment.

For subjects ``MA02CA`` and ``PE10MI`` no eyetracker samples were recorded. Only god knows why
