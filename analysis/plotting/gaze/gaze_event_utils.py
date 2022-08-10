from pylink import *


def get_blinks(events):
    """ Returns two lists of blinks in the form [start_time, end_time] """
    blink_starts = []
    blink_ends = []

    print('STARTBLINK', STARTBLINK)
    print('ENDBLINK', ENDBLINK)

    for e in events:
        print(e[1])
        if e[1] == STARTBLINK:
            blink_starts.append(e[0])
        if e[1] == ENDBLINK:
            blink_ends.append(e[0])

    return blink_starts, blink_ends


def get_fixations(events):
    """ Returns two lists of fixations in the form [start_time, end_time] """
    fix_starts = []
    fix_ends = []

    for e in events:
        if e[1] == STARTFIX:
            fix_starts.append(e[0])
        if e[1] == ENDFIX:
            fix_ends.append(e[0])

    return fix_starts, fix_ends


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
