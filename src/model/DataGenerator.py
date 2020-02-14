from os import path
from glob import glob

import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, RobustScaler
from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE


class DataGenerator(keras.utils.Sequence):

    def __init__(self, ID_list, labels):
        """
        Input pipeline for data in both pretraining and finetuning phases.

        These two pipelines are needed because the nature of training (and testing) is
        different on the two submodels trained by this model. The representation learner
        is trained with batches of signal epochs (usually 30s of signal per signal epoch)
        and does not require that the signals come from the same overall sequence, and thus
        can be shuffled when building the input from tfrecords. The sequential learner must
        learn on an entire signal sequence (usually per file or per sample) and is reset
        between sequences, therefore each file is read sequentially and in its entirety
        before generating the next set of training examples.
        """
        self.batch_size = FLAGS.batch_size
        self.list_IDs = ID_list
        self.labels = labels
        self.dim = FLAGS.s_per_epoch * EFFECTIVE_SAMPLE_RATE
        self.shuffle = True
        self.scaler = StandardScaler() #KBinsDiscretizer(n_bins=5, encode='ordinal')
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: epoch index
        :return: batch (X, y)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, batch_inds):
        """
        Generates data containing batch_size samples
        :param list_IDs_temp:
        :return: batch
        """
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, k in enumerate(batch_inds):
            # Store sample
            data = np.load(self.list_IDs[k])
            epoch = np.reshape(data['x'], (-1, 1))
            self.scaler.partial_fit(epoch)
            # epoch = np.fft.fft(epoch).real
            epoch = self.scaler.transform(epoch)

            if FLAGS.downsample_rate > 1:
                X[i,:] = np.reshape(epoch, (-1, self.dim, FLAGS.downsample_rate, 1)).mean(axis=2)
            else:
                X[i,:] = epoch

            # Store class
            y[i] = data['y'] - 1

        return X, keras.utils.to_categorical(y, num_classes=5)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return: None
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
