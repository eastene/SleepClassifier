import tensorflow as tf
import tensorflow.keras as keras


class RepresentationLearner(keras.Model):

    def __init__(self, sampling_rate=200, use_dp=False, use_bn=False):
        super(RepresentationLearner, self).__init__(name='rl')

        self.sampling_rate = sampling_rate
        self.use_dp = use_dp
        self.use_bn = use_bn

        self.top_layer_filters = 32
        self.hidden_layer_filters = 64

        """
        TEMPORAL FEATURIZER
        """

        self.conv_1_large = keras.layers.Conv1D(
            filters=self.top_layer_filters,
            kernel_size=self.sampling_rate // 2,
            strides=self.sampling_rate // 4,
            activation='relu',
            padding='SAME',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )

        if use_bn:
            self.batch_norm_large_1 = keras.layers.BatchNormalization(epsilon=1e-5)

        self.pool_1_large = keras.layers.MaxPool1D(pool_size=4, strides=4)

        # Dropout
        if self.use_dp:
            self.dropout_large = keras.layers.Dropout(rate=0.5)

        # Conv Layer 2
        self.conv_2_large = keras.layers.Conv1D(
            filters=self.hidden_layer_filters,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='SAME',
        )

        if use_bn:
            self.batch_norm_large_2 = keras.layers.BatchNormalization()

        # Conv Layer 3
        self.conv_3_large = keras.layers.Conv1D(
            filters=self.hidden_layer_filters,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        if use_bn:
            self.batch_norm_large_3 = keras.layers.BatchNormalization()

        # Conv Layer 4
        self.conv_4_large = keras.layers.Conv1D(
            filters=self.hidden_layer_filters,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        # Max Pool Layer 2
        self.pool_2_large = keras.layers.MaxPool1D(pool_size=2, strides=2)

        self.flatten_large = keras.layers.Flatten()

        """
        FREQUENCY FEATURIZER
        """

        self.conv_1_small = keras.layers.Conv1D(
            filters=self.top_layer_filters,
            kernel_size=self.sampling_rate // 8,
            strides=self.sampling_rate // 16,
            activation='relu',
            padding='SAME',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )

        if use_bn:
            self.batch_norm_small_1 = keras.layers.BatchNormalization(epsilon=1e-5)

        self.pool_1_small = keras.layers.MaxPool1D(pool_size=8, strides=8)

        # Dropout
        if self.use_dp:
            self.dropout_small = keras.layers.Dropout(rate=0.5)

        # Conv Layer 2
        self.conv_2_small = keras.layers.Conv1D(
            filters=self.hidden_layer_filters,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='SAME',
        )

        if use_bn:
            self.batch_norm_small_2 = keras.layers.BatchNormalization()

        # Conv Layer 3
        self.conv_3_small = keras.layers.Conv1D(
            filters=self.hidden_layer_filters,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        if use_bn:
            self.batch_norm_small_3 = keras.layers.BatchNormalization()

        # Conv Layer 4
        self.conv_4_small = keras.layers.Conv1D(
            filters=self.hidden_layer_filters,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='SAME'
        )

        # Max Pool Layer 2
        self.pool_2_small = keras.layers.MaxPool1D(pool_size=4, strides=4)

        self.flatten_small = keras.layers.Flatten()

        self.concat = keras.layers.Concatenate()

    def call(self, inputs):
        x1 = self.conv_1_large(inputs)
        if self.use_bn:
            x1 = self.batch_norm_large_1(x1)
        x1 = self.pool_1_large(x1)
        if self.use_dp:
            x1 = self.dropout_large(x1)
        x1 = self.conv_2_large(x1)
        if self.use_bn:
            x1 = self.batch_norm_large_2(x1)
        x1 = self.conv_3_large(x1)
        if self.use_bn:
            x1 = self.batch_norm_large_3(x1)
        x1 = self.conv_4_large(x1)
        x1 = self.flatten_large(x1)

        x2 = self.conv_1_small(inputs)
        if self.use_bn:
            x2 = self.batch_norm_small_1(x2)
        x2 = self.pool_1_small(x2)
        if self.use_dp:
            x2 = self.dropout_small(x2)
        x2 = self.conv_2_small(x2)
        if self.use_bn:
            x2 = self.batch_norm_small_2(x2)
        x2 = self.conv_3_small(x2)
        if self.use_bn:
            x2 = self.batch_norm_small_3(x2)
        x2 = self.conv_4_small(x2)
        x2 = self.flatten_small(x2)

        return self.concat([x1, x2])
