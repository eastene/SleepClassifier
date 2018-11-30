import numpy as np
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
        sizes = []
        sampling_rate = 0.0
        for f in files:
            data = np.loadtxt(f, dtype=np.float32, delimiter=',')
            print(data.shape)
            X.append(data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch])
            Y.append(data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch].astype(dtype=np.int64) - 1)
            sizes.append(data.shape[0])

        X, Y = np.vstack(X), np.hstack(Y)
        if (FLAGS.oversample):
            print("Pre Oversampling Label Counts {}".format(np.bincount(Y)))
            ros = RandomOverSampler()
            X_tmp, Y_tmp = ros.fit_sample(X, Y)
            print("Post Oversampling Label Counts {}".format(np.bincount(Y_tmp)))
        else:
            X_tmp, Y_tmp = X, Y

        print("Writing to tfrecord...", end=" ")
        offset = 0
        for i, f in enumerate(files):
            out_str = f + "_oversampled" if FLAGS.oversample else ""
            with tf.python_io.TFRecordWriter(out_str + '.tfrecord') as tfwriter:
                for j in range(sizes[i]):
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'signal': tf.train.Feature(
                                        float_list=tf.train.FloatList(value=X_tmp[offset + j, :])),
                                    'label': tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[Y_tmp[offset + j]]))
                                }
                            )
                        )
                        tfwriter.write(example.SerializeToString())
            offset += sizes[i]
        print("Done.")
