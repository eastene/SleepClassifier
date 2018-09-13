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

X, Y = np.vstack(X), np.hstack(Y)
print("Pre Oversampling Label Counts {}".format(np.bincount(Y)))
smt = RandomOverSampler()
X_os, Y_os = smt.fit_sample(X, Y)
print("Post Oversampling Label Counts {}".format(np.bincount(Y_os)))

max_examples_per_file = 2048
num_files = X_os.shape[0] // max_examples_per_file
for i in range(num_files):
    with tf.python_io.TFRecordWriter(str(i) + '_part.tfrecord') as tfwriter:
        for j in range(max_examples_per_file):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'signal': tf.train.Feature(float_list=tf.train.FloatList(value=X_os[i*max_examples_per_file + j])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[Y_os[i*max_examples_per_file + j]])),
                        'sampling_rate': tf.train.Feature(float_list=tf.train.FloatList(value=[sampling_rate]))
                    }
                )
            )

            tfwriter.write(example.SerializeToString())