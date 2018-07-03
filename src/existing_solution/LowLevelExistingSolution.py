import tensorflow as tf
import numpy as np
import glob

from tensorflow.contrib.layers import l2_regularizer
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler


tf.logging.set_verbosity(tf.logging.INFO)


def load_data():
    train_files = glob.glob('../../data/existing_solution/prepared_data/SC4*.npz')
    X = []
    Y = []
    sampling_rate = 0

    for f in train_files:
        data = np.load(f)
        X.append(data['x'])
        Y.append(data['y'])
        sampling_rate = data['fs']

    return np.vstack(X), np.hstack(Y), sampling_rate


def representation_learner(data, sampling_rate, mode, use_small_filter=True):
    """
    CNN with small filter size
    :param data: EEG signal input
    :param labels: Sleep stage labels
    :param sampling_rate: rate at which the EEG input is sampled
    :param mode: tensorflow mode (see tf.estimator.ModeKeys)
    :param use_small_filter: if true, sets filter size to sampling_rate / 2, else sets to sampling_rate * 4
    :return: pool_2: output of 2nd pooling layer
    """

    # Set Hyper Parameters Based on Filter Size (small vs large)
    conv_1_filter_size = int(sampling_rate / 2) if (use_small_filter) else int(sampling_rate * 4)
    conv_1_stride = int(sampling_rate / 16) if (use_small_filter) else int(sampling_rate / 2)
    max_pool_size = 8 if (use_small_filter) else 4
    pool_stride = 8 if (use_small_filter) else 4
    other_conv_size = 8 if (use_small_filter) else 6

    # Set Names for Export
    conv_1_name = "conv_1_small" if use_small_filter else "conv_1_large"
    conv_2_name = "conv_2_small" if use_small_filter else "conv_2_large"
    conv_3_name = "conv_3_small" if use_small_filter else "conv_3_large"
    conv_4_name = "conv_4_small" if use_small_filter else "conv_4_large"

    # Input layer
    # reshape EEG signal input to (# examples, # samples per 30s example), makes #-example 1D vectors
    input_layer = tf.reshape(data, [-1, 3000, 1])

    # Conv Layer 1
    batch_normalizer = tf.layers.batch_normalization(input_layer,epsilon=1e-5)
    l2_regulizer = l2_regularizer(0.001)
    conv_1 = tf.layers.conv1d(
        inputs=batch_normalizer,
        filters=64,
        kernel_size=conv_1_filter_size,
        strides=conv_1_stride,
        activation=tf.nn.relu,
        padding='SAME',
        kernel_regularizer=l2_regulizer,
        name=conv_1_name,
        reuse=tf.AUTO_REUSE
    )

    # Max Pool Layer 1
    pool_1 = tf.layers.max_pooling1d(inputs=conv_1, pool_size=max_pool_size, strides=pool_stride)

    # Dropout
    dropout = tf.layers.dropout(inputs=pool_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Conv Layer 2
    conv_2 = tf.layers.conv1d(
        inputs=dropout,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu,
        padding='SAME',
        name=conv_2_name,
        reuse=tf.AUTO_REUSE
    )

    # Conv Layer 3
    conv_3 = tf.layers.conv1d(
        inputs=conv_2,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu,
        padding='SAME',
        name=conv_3_name,
        reuse=tf.AUTO_REUSE
    )

    # Conv Layer 4
    conv_4 = tf.layers.conv1d(
        inputs=conv_3,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu,
        padding='SAME',
        name=conv_4_name,
        reuse=tf.AUTO_REUSE
    )

    # Max Pool Layer 2
    pool_2 = tf.layers.max_pooling1d(inputs=conv_4, pool_size=max_pool_size // 2, strides=pool_stride // 2)

    return pool_2


def pretrain(mode=tf.estimator.ModeKeys.TRAIN):
    # hyper-parameters
    n_folds = 20
    fold_test_split = 0.1
    epochs = 100
    learn_rate = 0.0001
    batch_size = 100
    sampling_rate = 100

    # inputs
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int32)

    # define model
    input_layer = tf.reshape(x, [-1, 3000, 1])
    small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True)
    large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False)
    cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)
    logits = tf.layers.dense(inputs=flat_layer, units=5, name='logits')

    # define model trainer
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    optimiser = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999)
    train_op = optimiser.minimize(loss, global_step=tf.train.get_global_step())

    # define model evaluator
    correct_classes = tf.equal(tf.cast(y, tf.int64), tf.argmax(logits, axis=1))
    eval_op = tf.reduce_mean(tf.cast(correct_classes, tf.float32))

    # define model predictor
    pred_classes = tf.argmax(logits, axis=1)

    # define initialisation operator
    init_op = tf.global_variables_initializer()

    # batches
    batch = tf.train.batch([image, label], batch_size=100)

    with tf.Session() as sess:
        sess.run(init_op)

        data, labels, fs = load_data()
        r = data.shape[0]
        c = data.shape[1]

        # Save some test data for testing after cross validation
        res_data, test_data, res_labels, test_labels = train_test_split(np.reshape(data, (r, c)), labels, test_size=0.1)

        print("Pre Oversampling Label Counts {}".format(np.bincount(labels)))
        ros = RandomOverSampler()
        os_data, os_labels = ros.fit_sample(X=res_data, y=res_labels)
        print("Post Oversampling Label Counts {}".format(np.bincount(os_labels)))

        folds = KFold(n_splits=n_folds)

        for train_idx, eval_idx in folds.split(os_data):

            train_data = os_data[train_idx]
            train_labels = os_labels[train_idx]
            eval_data = os_data[eval_idx]
            eval_labels = os_labels[eval_idx]

            ts = test_data.shape[0]  # test data size
            es = eval_data.shape[0]  # eval data size

            for epoch in range(epochs):
                # Train Loop
                """avg_cost = 0
                for index, offset in enumerate(range(0, train_data.shape[0], batch_size)):
                    train_data_batch = train_data[offset:offset + batch_size]
                    train_label_batch = train_labels[offset:offset + batch_size]

                    feed_dict = {
                        x: train_data_batch,
                        y: train_label_batch
                    }

                    _, c = sess.run(
                        [train_op, loss],
                        feed_dict=feed_dict
                    )
                """
                print(tf.train.batch([train_data, train_labels], batch_size=100, num_threads=8))
                _, c = sess.run(
                    [train_op, loss],
                    feed_dict=tf.train.batch({x: train_data, y: train_labels}, batch_size=100, num_threads=8)
                )

                avg_cost += c / train_data.shape[0]

                if epoch % 10 == 0:
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

                # Eval Loop
                for index, offset in enumerate(range(0, eval_data.shape[0], batch_size)):
                    eval_data_batch = eval_data[offset:offset + batch_size]
                    eval_label_batch = eval_labels[offset:offset + batch_size]

                    feed_dict = {
                        x: eval_data_batch,
                        y: eval_label_batch
                    }

                    acc = sess.run(
                        eval_op,
                        feed_dict=feed_dict
                    )

        # Test after all epochs
        predictions = sess.run(pred_classes, {x:test_data, y:test_labels})
        con_mat = tf.confusion_matrix(labels=list(test_labels), predictions=list(predictions))
        print("Results of Pretraining Feature Representaion:")
        print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))


def main(unused_argv):
    pretrain(tf.estimator.ModeKeys.TRAIN)


if __name__ == "__main__":
    tf.app.run()