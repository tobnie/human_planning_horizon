import numpy as np
import itertools


def generate_possible_states(n):
    """ Generates all possible states (occupied / not occupied) for a given n, resulting in a
    n x n matrix with the player in the middle.
    """
    possible_states_flat = list(set(itertools.combinations([0, 1] * 8, n ** 2 - 1)))

    # turn into nxn fields
    possible_states = []
    for state in sorted(possible_states_flat):
        state = np.array(state)
        test = (n ** 2 - 1) // 2
        state = np.insert(state, (n ** 2 - 1) // 2, -1)
        possible_states.append(np.reshape(state, (n, n)))

    possible_states = sorted(possible_states, key=lambda x: str(x))
    return possible_states