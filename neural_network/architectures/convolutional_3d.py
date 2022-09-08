from neural_network.architectures.base_network import NeuralNetwork
from tensorflow import keras

nn_configuration = {
    'epochs': 50,  # number of epochs
    'batch_size': 32,  # size of the batch
    'verbose': 1,  # set the training phase as verbose
    'optimizer': keras.optimizers.Adam(clipvalue=1.0),  # optimizer
    'metrics': ["root_mean_squared_error"],
    'loss': 'mean_squared_error',  # loss
    'val_split': 0.2,  # validation split: percentage of the training data used for evaluating the loss function
    'input_shape': (15, 20, 3),
    'n_output': 2  # number of outputs = x and y
}


class ConvNetwork3D(NeuralNetwork):

    def __init__(self, name, percent_train=0.8, configuration=None):
        X_path = '../data/input_deep_fm.npz'
        y_path = '../data/output.npz'
        super().__init__(name, percent_train, configuration)
        self._load_data(X_path, y_path)

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(input_shape=self.config['input_shape'], filters=8, kernel_size=3,
                                strides=1, padding='same', activation='relu', name='Conv1_In'),
            keras.layers.MaxPooling2D(pool_size=2, strides=None, padding="valid"),
            keras.layers.Flatten(),
            keras.layers.Dense(20, name='Dense1'),  # TODO activation='relu'),
            keras.layers.Dense(2, name='Dense2_Out')
        ])

        print(f'Created model for {self.name}:')
        print(self.model.summary())
