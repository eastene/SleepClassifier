from os import path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from src.model.DataGenerator import DataGenerator
from data.DataPrepper import DataPrepper
from src.model.RepresentationLearner import RepresentationLearner
from src.model.SequenceResidualLearner import SequenceResidualLearner
from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE


class DeepSleepNet:

    def __init__(self):
        # self.prep = DataPrepper()
        # self.prep.mat2npy(files=glob(path.join(FLAGS.data_dir, "*.mat")))

        # hyper-parameters
        # self.n_folds = 20
        self.sampling_rate = EFFECTIVE_SAMPLE_RATE
        self.n_classes = 5

        # data parsing
        self.epoch_files = glob(path.join(FLAGS.data_dir, "*", "*.npz"))
        print("Generating labels")
        self.labels = np.squeeze(np.array([np.load(f)['y'].astype("int") for f in self.epoch_files]))
        train_split = int(len(self.epoch_files) * (1 - FLAGS.test_split))
        val_split = train_split - int(train_split * FLAGS.val_split)
        self.train_epochs = self.epoch_files[:val_split]
        self.val_epochs = self.epoch_files[val_split:train_split]
        self.test_epochs = self.epoch_files[train_split:]
        self.train_gen = DataGenerator(self.train_epochs, self.labels)
        self.val_gen = DataGenerator(self.train_epochs, self.labels)
        self.test_gen = DataGenerator(self.train_epochs, self.labels)

    def train(self):
        weights = class_weight.compute_class_weight('balanced', np.unique(self.labels), self.labels)

        saver = keras.callbacks.ModelCheckpoint(filepath=path.join(FLAGS.checkpoint_dir, "rep_learn.h5"))
        stopper = keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=5)

        print("Pretraining Featurizer")
        model_rl = keras.Sequential([
            RepresentationLearner(self.sampling_rate, use_dropout=True),
            keras.layers.Dense(units=self.n_classes, activation="softmax")
        ])
        model_rl.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model_rl.fit_generator(generator=self.train_gen, validation_data=self.val_gen, class_weight=weights,
                               callbacks=[saver, stopper])
        model_rl.evaluate_generator(generator=self.test_gen)


if __name__ == "__main__":
    dn = DeepSleepNet()
    dn.train()
