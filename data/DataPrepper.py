import numpy as np
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from src.model.flags import FLAGS


class DataPrepper:

    def __init__(self):
        self.files = []

    def convert2tfrecord(self, files):
        # load all data
        X = []
        Y = []
        sizes = []
        sampling_rate = 0.0

        self.files = files
        for f in files:
            data = np.genfromtxt(f, dtype=np.float64, delimiter=',', filling_values=[0])
            X.append(data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch])
            Y.append(data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch].astype(dtype=np.int64) - 1)
            sizes.append(data.shape[0])

        X_s, Y_s = np.vstack(X), np.hstack(Y)
        X_s[X_s == np.inf] = 0
        X_s[X_s == -np.inf] = 0
        if (FLAGS.oversample):
            print("Pre Oversampling Label Counts {}".format(np.bincount(Y_s)))
            ros = RandomOverSampler()
            X_tmp, Y_tmp = ros.fit_sample(X_s, Y_s)
            print("Post Oversampling Label Counts {}".format(np.bincount(Y_tmp)))
        else:
            X_tmp, Y_tmp = X_s, Y_s

        print("Writing to tfrecord... [", end=" ")
        offset = 0
        for i, f in enumerate(files):
            out_str = f + "_oversampled" if FLAGS.oversample else ""
            with tf.python_io.TFRecordWriter(out_str + '.tfrecord') as tfwriter:
                for j in range(sizes[i]):
                    x = X_tmp[offset + j, :]
                    if FLAGS.resample_rate > 0:
                        x = x.reshape(-1, FLAGS.resample_rate).mean(axis=1)
                    y = Y_tmp[offset + j]
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'signal': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=x)),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[y]))
                            }
                        )
                    )
                    tfwriter.write(example.SerializeToString())
            offset += sizes[i]
        print("Done.")

#TODO implement buffer
    def read_seq_files_2_buffer(self, start_idx, buff_pos, buffer):
        _buffer = buffer

        # buffer larger than number of files, read
        if len(self.files) < FLAGS.seq_buff_size:

            if len(buffer) == 0:
                for f in files:
                    data = np.loadtxt(f, dtype=np.float32, delimiter=',')
            return buffer
