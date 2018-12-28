import tensorflow as tf
import numpy as np
from glob import glob
import ntpath
import os

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

DATA_DIR = '/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_data/'
FILE_PATTERN = DATA_DIR + '*.npz'

npz_files = glob(FILE_PATTERN)

# load all data
X = []
Y = []
sampling_rate = 0.0
for f in npz_files:
    data = np.load(f)
    assert(data['x'].shape[0] == data['y'].shape[0])

    X.append(np.squeeze(data['x']))
    Y.append(data['y'])
    sampling_rate = data['fs']

X_arr, Y_arr = np.vstack(X), np.hstack(Y)
ros = RandomOverSampler()
x_out, y_out = ros.fit_sample(X_arr, Y_arr)

new_file_size = x_out.shape[0] // len(npz_files)
leftovers = x_out.shape[0] % len(npz_files)
for i, f in enumerate(npz_files):
    with tf.python_io.TFRecordWriter(os.path.splitext(f)[0] + '.tfrecords') as tfwriter:
        iter_range = new_file_size if i < len(npz_files) - 1 else new_file_size + leftovers
        for j in range(iter_range):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'signal': tf.train.Feature(float_list=tf.train.FloatList(value=x_out[i * new_file_size + j, :])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_out[i * new_file_size + j]])),
                        'sampling_rate': tf.train.Feature(float_list=tf.train.FloatList(value=[sampling_rate]))
                    }
                )
            )

            tfwriter.write(example.SerializeToString())