import os
import numpy as np
import tensorflow as tf

from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE, META_INFO_FNAME

class DataPrepper:

    def __init__(self):
        self.csv_files = []
        self.meta = {
            'fs': FLAGS.sampling_rate,
            'rs': FLAGS.resample_rate,
            'rows': 0,
            'nfiles': 0
        }

    def load_meta_info(self):
        # checks that existing metainfo matches flags passed to program
        temp_meta = np.load(os.path.join(FLAGS.meta_dir, META_INFO_FNAME))
        temp_meta = temp_meta['a'][()]

        if self.meta['fs'] * self.meta['rs'] != temp_meta['fs'] * temp_meta['rs']:
            print("Error: Effective sampling rate of data does not match rate indicated by flags!")
            print("Exiting.")
            exit(1)
        self.meta = temp_meta

    def csv2npz(self, files):
        # convert csv files to npz for more efficient access
        self.meta['nfiles'] += len(files)
        rows = self.meta['rows']
        for f in files:
            data = np.genfromtxt(f, dtype=np.float32, delimiter=',', filling_values=[0])
            x = data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch]
            y = data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch].astype(dtype=np.int64) - 1
            if FLAGS.resample_rate > 0:
                x = x.reshape(x.shape[0], -1, FLAGS.resample_rate).mean(axis=2)
            np.savez_compressed(os.path.join(FLAGS.seq_dir, os.path.splitext(os.path.basename(f))[0]), x=x, y=y,
                                fs=EFFECTIVE_SAMPLE_RATE)
            rows += x.shape[0]
        self.meta['rows'] = rows
        np.savez_compressed(os.path.join(FLAGS.meta_dir, os.path.splitext(META_INFO_FNAME)[0]), a=self.meta)

    def find_missing(self, file_set1, file_set2):
        """
        Finds files not present in set1 that are in set2 (e.g. set_1 \ set_2)
        :param file_set1: list of file names
        :param file_set2: list of file names
        :return: list of missing file paths
        """
        # assumes filename up to first . will be unique (allows .'s before extension to be file descriptors)
        return [f1 for f1 in file_set1 if
         all(os.path.basename(f1).split(".")[0] not in os.path.basename(f2).split(".")[0] for f2 in file_set2)]

    def get_rows(self):
        return self.meta['rows']

    def load_epochs(self, npz_files, train=True):
        X = []
        Y = []
        if train:
            print("Training Files:")
        else:
            print("Eval Files:")
        for f in npz_files:
            print(f)
            data = np.load(f)
            x = data['x']
            y = data['y']
            if FLAGS.resample_rate > 0:
                x = x.reshape(x.shape[0], -1, FLAGS.resample_rate).mean(axis=2)
            X.append(x)
            Y.append(y.astype(dtype=np.int64))

        X_s, Y_s = np.vstack(X), np.hstack(Y)
        X_s[X_s == np.inf] = 0
        X_s[X_s == -np.inf] = 0

        # only oversample training set (no need to oversample eval)
        if FLAGS.oversample and train:
            counts = np.bincount(Y_s)
            print("Pre Oversampling Label Counts {}".format(counts))
            X_os = []
            Y_os = []
            # random oversampling (make all classes equal to majority size)
            max_count = max(counts)
            for i, count in enumerate(counts):
                inds = np.random.choice(count, size=max_count, replace=True)
                Y_os.append(Y_s[Y_s == i][inds])
                X_os.append(X_s[Y_s == i][inds])
            x_out, y_out = np.vstack(X_os), np.hstack(Y_os)
            print("Post Oversampling Label Counts {}".format(np.bincount(y_out)))
        else:
            x_out, y_out = X_s, Y_s

        return list(zip(x_out, y_out))
