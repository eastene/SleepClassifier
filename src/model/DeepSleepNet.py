import os
from glob import glob

import numpy as np
# import matplotlib.pyplot as plt
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
        # self.prep.mat2npy(files=glob(os.path.join(FLAGS.data_dir, "*.mat")))

        # hyper-parameters
        # self.n_folds = 20
        self.sampling_rate = EFFECTIVE_SAMPLE_RATE
        self.n_classes = 5

        # data parsing
        # self.data_dirs = glob(os.path.join(FLAGS.data_dir, "Staging_Data_GT*/"))
        # self.data_dirs.sort()
        # self.epoch_files = [glob(os.path.join(dd, "epoch*.npy")) for dd in self.data_dirs]
        # for l in self.epoch_files:
        #     l.sort()
        self.epoch_files = glob(os.path.join(FLAGS.data_dir, "Staging_Data_GT*/", "*.npz")) #[ef for lists in self.epoch_files for ef in lists]
        # print(self.epoch_files)
        # self.label_files = [glob(os.path.join(dd, "labels.npy")) for dd in self.data_dirs]
        # self.label_files = [lf for lists in self.label_files for lf in lists]
        print("Generating labels")
        self.labels = np.hstack([np.squeeze(np.load(f)['y']) - 1 for f in self.epoch_files])

        self.train_split = int(len(self.epoch_files) * (1 - FLAGS.test_split))
        self.val_split = self.train_split - int(self.train_split * FLAGS.val_split)
        self.train_gen = DataGenerator(self.epoch_files[:self.val_split], self.labels[:self.val_split])
        self.val_gen = DataGenerator(self.epoch_files[self.val_split:self.train_split], self.labels[self.val_split:self.train_split])
        self.test_gen = DataGenerator(self.epoch_files[self.train_split:], self.labels[self.train_split:])

    def train(self):
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        weights = class_weight.compute_class_weight('balanced', np.unique(self.labels), self.labels)
        class_weights = {i: weights[i] for i in range(self.n_classes)}

        saver_rl = keras.callbacks.ModelCheckpoint(filepath=os.path.join(FLAGS.checkpoint_dir, "rep_learn.h5"),
                                                save_weights_only=True, save_best_only=True)
        stopper_rl = keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15, verbose=1)

        print("Pretraining Featurizer")
        model_rl = keras.Sequential([
            keras.layers.Input(shape=(self.sampling_rate * 30, 1)),
            RepresentationLearner(self.sampling_rate),
            keras.layers.Dense(units=self.n_classes, activation="softmax")
        ])
        model_rl.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                         metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])
        model_rl.summary()
        model_rl.fit(self.train_gen, validation_data=self.val_gen, class_weight=class_weights,
                     callbacks=[stopper_rl, saver_rl], epochs=FLAGS.num_epochs_pretrain, verbose=1)
        model_rl.evaluate(self.test_gen, verbose=1)

        # print("Finetuning model")
        # saver_sl = keras.callbacks.ModelCheckpoint(filepath=os.path.join(FLAGS.checkpoint_dir, "seq_learn.h5"),
        #                                         save_weights_only=True, save_best_only=True)
        # stopper_sl = keras.callbacks.EarlyStopping(monitor='val_recall', restore_best_weights=True, patience=5, verbose=1)
        # model_sl = keras.Sequential([
        #     keras.layers.Input(shape=(self.sampling_rate * 30, 1)),
        #     RepresentationLearner(self.sampling_rate),
        #     SequenceResidualLearner(),
        #     keras.layers.Dense(units=self.n_classes, activation="softmax")
        # ])
        # model_sl.compile(optimizer="rmsprop", loss="categorical_crossentropy",
        #                  metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        # model_sl.summary()
        # # model_sl.fit(self.train_gen, validation_data=self.val_gen, class_weight=class_weights,
        # #              callbacks=[stopper_sl, saver_sl], epochs=FLAGS.num_epochs_pretrain, verbose=1)
        # # model_sl.evaluate(self.test_gen, verbose=1)
        #
        labs = np.argmax(model_rl.predict(self.test_gen), axis=1)
        print(np.bincount(labs))
        # np.savez("labels.npz", pred=labs, gt=self.labels[self.train_split:])


if __name__ == "__main__":
    dn = DeepSleepNet()
    dn.train()
