import tensorflow as tf
import tensorflow.keras as keras


class RepresentationLearner(keras.Model):

    def __init__(self, sampling_rate=1000, use_dropout=False):
        super(RepresentationLearner, self).__init__(name='rl')

        self.sampling_rate = sampling_rate
        self.use_dp = use_dropout

        self.batch_norm = keras.layers.BatchNormalization()

        """
        TEMPORAL FEATURIZER
        """

        self.conv_1_large = keras.layers.Conv1D(
            filters=64,
            kernel_size=self.sampling_rate * 4,
            strides=self.sampling_rate // 2,
            activation='relu',
            padding='SAME',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )

        self.pool_1_large = keras.layers.MaxPool1D(pool_size=4, strides=4)

        # Dropout
        if self.use_dp:
            self.dropout_large = keras.layers.Dropout(rate=0.5)

        # Conv Layer 2
        self.conv_2_large = keras.layers.Conv1D(
            filters=128,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='SAME',
        )

        # Conv Layer 3
        self.conv_3_large = keras.layers.Conv1D(
            filters=128,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        # Conv Layer 4
        self.conv_4_large = keras.layers.Conv1D(
            filters=128,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        # Max Pool Layer 2
        self.pool_2_large = keras.layers.MaxPool1D(pool_size=2, strides=2)

        """
        FREQUENCY FEATURIZER
        """

        self.conv_1_small = keras.layers.Conv1D(
            filters=64,
            kernel_size=self.sampling_rate // 2,
            strides=self.sampling_rate // 16,
            activation='relu',
            padding='SAME',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )

        self.pool_1_small = keras.layers.MaxPool1D(pool_size=8, strides=8)

        # Dropout
        if self.use_dp:
            self.dropout_small = keras.layers.Dropout(rate=0.5)

        # Conv Layer 2
        self.conv_2_small = keras.layers.Conv1D(
            filters=128,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='SAME',
        )

        # Conv Layer 3
        self.conv_3_small = keras.layers.Conv1D(
            filters=128,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        # Conv Layer 4
        self.conv_4_small = keras.layers.Conv1D(
            filters=128,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        # Max Pool Layer 2
        self.pool_2_small = keras.layers.MaxPool1D(pool_size=4, strides=4)

        self.flatten = keras.layers.Flatten()

        self.concat = keras.layers.Concatenate()

    def call(self, inputs):
        x = self.batch_norm(inputs)

        x1 = self.conv_1_large(x)
        x1 = self.pool_1_large(x1)
        if self.use_dp:
            x1 = self.dropout_large(x1)
        x1 = self.conv_2_large(x1)
        x1 = self.conv_3_large(x1)
        x1 = self.conv_4_large(x1)
        x1 = self.flatten(x1)

        x2 = self.conv_1_small(x)
        x2 = self.pool_1_small(x2)
        if self.use_dp:
            x2 = self.dropout_small(x2)
        x2 = self.conv_2_small(x2)
        x2 = self.conv_3_small(x2)
        x2 = self.conv_4_small(x2)
        x2 = self.flatten(x2)

        return self.concat([x1, x2])
