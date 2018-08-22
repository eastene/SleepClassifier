import tensorflow as tf
import numpy as np
from glob import glob
import ntpath
import os

DATA_DIR = '/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_data/'
FILE_PATTERN = DATA_DIR + '*.npz'

npz_files = glob(FILE_PATTERN)

for f in npz_files:
    data = np.load(f)
    assert(data['x'].shape[0] == data['y'].shape[0])

    # generate new filename with .tfrecords extension
    filename = ntpath.basename(f)
    base, ext = os.path.splitext(filename)
    print("Writing " + base + '.tfrecord')

    signals = data['x']
    labels = data['y']

    with tf.python_io.TFRecordWriter(base + '.tfrecord') as tfwriter:

        for i in range(data['x'].shape[0]):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'signal': tf.train.Feature(float_list=tf.train.FloatList(value=signals[i])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                        'sampling_rate': tf.train.Feature(float_list=tf.train.FloatList(value=[data['fs']]))
                    }
                )
            )

            tfwriter.write(example.SerializeToString())