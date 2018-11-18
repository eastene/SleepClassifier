import tensorflow as tf


class DataPrepper:

    def __init__(self):
        pass

    def convert2tfrecord(self, files):
        max_examples_per_file = 2048
        num_files = X_os.shape[0] // max_examples_per_file
        for i in range(num_files):
            with tf.python_io.TFRecordWriter(str(i) + '_part.tfrecord') as tfwriter:
                for j in range(max_examples_per_file):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'signal': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=X_os[i * max_examples_per_file + j])),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[Y_os[i * max_examples_per_file + j]]))
                            }
                        )
                    )

                    tfwriter.write(example.SerializeToString())