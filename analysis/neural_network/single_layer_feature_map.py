import numpy as np

from analysis.data_utils import read_data, read_subject_data
from data.save_data_compressed import create_feature_map_from_state


def create_state_from_str(string):
    return np.fromstring(string)


# player: 1, vehicle: 2, lilypad: 3
def create_single_layer_feature_map_from_state(state):
    # create 'deep' feature map
    normal_fm = create_feature_map_from_state(state)

    single_layer_fm = np.zeros_like(normal_fm[:, :, 0])

    # transform into feature_map where number encodes object instead of depth
    for depth_i in reversed(range(normal_fm.shape[-1])):
        obj_positions = np.where(normal_fm[:, :, depth_i] == 1)

        # by the following we can recognize collisions in a feature map by a value > 10
        # TODO do we want it that way? we could keep it. if we discard it, the player value is written last, such that the underlying collision is hidden
        single_layer_fm[obj_positions] = single_layer_fm[obj_positions] * 10 + depth_i + 1

    return single_layer_fm

# df = read_subject_data('AN06AN')
# example_state = np.array(df['state'][0])
# fm_test = create_single_layer_feature_map_from_state(example_state)
# print(fm_test)
