from os import path

import tensorflow.keras as keras

from src.model.flags import FLAGS
from src.model.RepresentationLearner import RepresentationLearner


class SequenceResidualLearner(keras.Model):

    def __init__(self, use_dp=False, use_bn=False):
        # Initialize Rep Learner
        super(SequenceResidualLearner, self).__init__(name="sl")

        # Housekeeping Parameters
        self.seq_learn_dir = path.join(FLAGS.checkpoint_dir, "seq_learn", "")

        # Hyperparameters
        self.lstm_size = 512
        self.use_dp = use_dp
        self.use_bn = use_bn

        self.prep = keras.layers.Reshape(target_shape=(FLAGS.sequence_length, -1))
        if self.use_dp:
            self.dp_1 = keras.layers.Dropout(rate=0.5)
        self.bd_lstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(units=self.lstm_size, return_sequences=True))
        if self.use_dp:
            self.dp_2 = keras.layers.Dropout(rate=0.5)
        self.bd_lstm_2 = keras.layers.Bidirectional(keras.layers.LSTM(units=self.lstm_size, return_sequences=True))
        if self.use_dp:
            self.dp_3 = keras.layers.Dropout(rate=0.5)
        self.lstm_out = keras.layers.Reshape(target_shape=(-1, 1024))

        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(epsilon=1e-5)
        self.shortcut = keras.layers.Dense(units=1024, activation='relu')

        self.add = keras.layers.Add()

    def call(self, inputs):
        x = self.prep(inputs)
        # if self.use_dp:
        #     x = self.dp_1(x)
        # x = self.bd_lstm_1(x)
        # if self.use_dp:
        #     x = self.dp_2(x)
        # x = self.bd_lstm_2(x)
        # if self.use_dp:
        #     x = self.dp_3(x)
        # x = self.lstm_out(x)

        x2 = self.shortcut(inputs)
        return self.add([x, x2])