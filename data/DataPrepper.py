import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from src.model.flags import FLAGS


class DataPrepper:

    def __init__(self):
        pass

    def convert2tfrecord(self, files):
        # load all data
        X = []
        Y = []
        sampling_rate = 0.0
        for f in files:
            data = pd.read_csv(f).values
            X.append(data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch])
            Y.append(data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch])

        X, Y = np.vstack(X), np.hstack(Y)
        if (FLAGS.oversample):
            print("Pre Oversampling Label Counts {}".format(np.bincount(Y.astype(dtype=np.int64))))
            ros = RandomOverSampler()
            X, Y = ros.fit_sample(X, Y)
            print("Post Oversampling Label Counts {}".format(np.bincount(Y.astype(dtype=np.int64))))

        print("Writing to tfrecord...", end=" ")
        max_examples_per_file = 2048
        num_files = X.shape[0] // max_examples_per_file
        for i in range(num_files):
            with tf.python_io.TFRecordWriter(str(i) + '_part.tfrecord') as tfwriter:
                for j in range(max_examples_per_file):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'signal': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=X[i * max_examples_per_file + j])),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[Y[i * max_examples_per_file + j]]))
                            }
                        )
                    )

                    tfwriter.write(example.SerializeToString())
        print("Done.")
