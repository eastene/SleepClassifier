import tensorflow as tf
import numpy as np
import glob

from tensorflow.contrib.layers import l2_regularizer
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 100, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 100, "size (in batches) of in-memory buffer to prefetch records before parsing")
tf.flags.DEFINE_integer("num_epochs_pretrain", 40, "number of epochs for pre-training")
tf.flags.DEFINE_integer("num_epochs_finetune", 1, "number of epochs for fine tuning")
tf.flags.DEFINE_string("checkpoint_dir_reuse", "/tmp/existing_model_reuse/",
                       "directory in which to save model parameters from pretraining that are reused in finetuning while training")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/existing_model/", "directory in which to save model parameters while training")

FLAGS = tf.flags.FLAGS
TRAIN_FILE_PATTERN = "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_tf/SC40*.tfrecord"
EVAL_FILE_PATTERN = "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_tf/SC41[0-7]*.tfrecord"
TEST_FILE_PATTERN = "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_tf/SC41[8-9]*.tfrecord"


"""
*
* vv INPUT PIPELINE
*
"""


def sleep_data_parse_fn(example):
    # format of each training example
    example_fmt = {
        "signal": tf.FixedLenFeature((1,3000), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64, -1),
        "sampling_rate": tf.FixedLenFeature((), tf.float32, 0.0)
    }

    parsed = tf.parse_single_example(example, example_fmt)

    return parsed['signal'], parsed['label'], parsed['sampling_rate']


def input_fn(file_pattern):
    files = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=False)

    # interleave reading of dataset for parallel I/O
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers
        )
    )

    dataset = dataset.cache()

    # shuffle data
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

    # parse the data and prepares the batches in parallel (helps most with larger batches)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            map_func=sleep_data_parse_fn, batch_size=FLAGS.batch_size
        )
    )

    # prefetch data so that the CPU can prepare the next batch(s) while the GPU trains
    # recommmend setting buffer size to number of training examples per training step
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    return dataset

"""
*
* ^^ INPUT PIPELINE
*
"""


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
    batch_normalizer = tf.layers.batch_normalization(input_layer, epsilon=1e-5)
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


def split_input(next_element):
    return next_element[0], next_element[1], next_element[2]


def pretrain(mode=tf.estimator.ModeKeys.TRAIN):
    # hyper-parameters
    n_folds = 20
    fold_test_split = 0.1
    learn_rate = 0.0001
    sampling_rate = 100.00

    # train batch iterator
    train_iter = input_fn(TRAIN_FILE_PATTERN).make_initializable_iterator()
    next_train_elem = train_iter.get_next()

    # eval batch iterator
    eval_iter = input_fn(EVAL_FILE_PATTERN).make_initializable_iterator()
    next_eval_elem = eval_iter.get_next()

    # inputs
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int32)

    # define model
    input_layer = tf.reshape(x, [-1, 3000, 1])
    #with tf.name_scope("REP_LEARN") as scope:
    # Shared parameters
    small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True)
    large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False)
    cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)
    logits = tf.layers.dense(inputs=flat_layer, units=5, name='logits', reuse=tf.AUTO_REUSE)

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

    # saver
    save_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize or restore
        if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
            saver.restore(sess, FLAGS.checkpoint_dir)
            print("Model restored.")
        else:
            sess.run(init_op)

        # PRETRAINING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            sess.run(train_iter.initializer)
            cost = 0.0
            n_batches = 0

            try:
                while True:
                    data = sess.run(next_train_elem)
                    feed_dict={
                        x: data[0],
                        y: data[1]
                    }

                    _, c = sess.run(
                        [train_op, loss],
                        feed_dict=feed_dict
                    )

                    cost += c
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                save_path = saver.save(sess, FLAGS.checkpoint_dir)
                print("Model saved in path: {}".format(save_path))

        # PRETRAINING EVAL LOOP
        m_tot = 0
        n_batches = 0
        #for epoch in range(FLAGS.num_epochs_pretrain):
        sess.run(eval_iter.initializer)

        # Evaluate
        try:
            while True:
                data = sess.run(next_eval_elem)
                feed_dict={
                    x: data[0],
                    y: data[1]
                }

                m = sess.run(
                    [eval_op],
                    feed_dict=feed_dict
                )
                n_batches += 1
                m_tot += m[0]

        except tf.errors.OutOfRangeError:
            pass  # reached end of epoch

        print("Pretraining Accuracy: {}".format(m_tot / n_batches))


    """
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
                avg_cost = 0
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
        """

def fine_tune(mode=tf.estimator.ModeKeys.TRAIN):
    # hyper-parameters
    n_folds = 20
    fold_test_split = 0.1
    learn_rate = 0.000001
    sampling_rate = 100.00

    # batch iterator
    dataset_iter = input_fn().make_initializable_iterator()
    next_elem = dataset_iter.get_next()

    # inputs
    # TODO change fs to sampling_rate
    x, y, fs = split_input(next_elem)

    # define model
    input_layer = tf.reshape(x, [-1, 3000, 1])
    small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True)
    large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False)
    cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)
    logits = tf.layers.dense(inputs=flat_layer, units=5, name='logits', reuse=tf.AUTO_REUSE)

    # define model trainer
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    optimiser = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, name="adam_finetune_rep")
    train_op = optimiser.minimize(loss, global_step=tf.train.get_global_step())

    # define model evaluator
    correct_classes = tf.equal(tf.cast(y, tf.int64), tf.argmax(logits, axis=1))
    eval_op = tf.reduce_mean(tf.cast(correct_classes, tf.float32))

    # define model predictor
    pred_classes = tf.argmax(logits, axis=1)

    # define initialisation operator
    init_op = tf.global_variables_initializer()

    # saver
    saver = tf.train.Saver()

    # FINETUNING LOOP
    with tf.Session() as sess:

        for epoch in range(FLAGS.num_epochs_pretrain):
            sess.run(dataset_iter.initializer)
            cost = 0.0
            n_batches = 0

            try:
                while True:
                    _, c = sess.run(
                        [train_op, loss]
                    )

                    cost += c
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                save_path = saver.save(sess, FLAGS.checkpoint_dir)
                print("Model saved in path: {}".format(save_path))


def main(unused_argv):
    pretrain(tf.estimator.ModeKeys.TRAIN)
    #fine_tune(tf.estimator.ModeKeys.TRAIN)


if __name__ == "__main__":
    tf.app.run()