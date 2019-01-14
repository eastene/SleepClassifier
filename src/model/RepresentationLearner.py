import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
from os import path

from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE


class RepresentationLearner:

    def __init__(self):
        # Housekeeping Parameters
        self.gen_dir = FLAGS.checkpoint_dir  # where to save whole model
        self.rep_learn_dir = path.join(FLAGS.checkpoint_dir, "rep_learn", "")  # where to save reusable layers
        self.sampling_rate = EFFECTIVE_SAMPLE_RATE

        # Hyperparameters
        self.mode = "TRAIN"  # default value, mutable, only used for dropout layers
        self.phase = "PRE"
        self.learning_rate = 0.0

        # Begin Define Model
        """
        Input Layer
        """
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.int32)
        self.input_layer = tf.reshape(self.x, [-1, self.sampling_rate * FLAGS.s_per_epoch, 1])

        # Shared Conv Layers
        self.batch_normalizer = tf.layers.batch_normalization(self.input_layer, epsilon=1e-5)
        self.l2_regulizer = l2_regularizer(0.001)

        """
        Temporal Convolutional Layers
        """
        # Conv Layer 1
        self.conv_1_large = tf.layers.conv1d(
            inputs=self.batch_normalizer,
            filters=64,
            kernel_size=self.sampling_rate * 4,
            strides=self.sampling_rate // 2,
            activation=tf.nn.relu,
            padding='SAME',
            kernel_regularizer=self.l2_regulizer,
            name="conv1_large",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 1
        self.pool_1_large = tf.layers.max_pooling1d(inputs=self.conv_1_large, pool_size=4, strides=4)

        # Dropout
        self.dropout_large = tf.layers.dropout(inputs=self.pool_1_large, rate=0.5, training=self.mode == "TRAIN")

        # Conv Layer 2
        self.conv_2_large = tf.layers.conv1d(
            inputs=self.dropout_large,
            filters=128,
            kernel_size=6,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv2_large",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 3
        self.conv_3_large = tf.layers.conv1d(
            inputs=self.conv_2_large,
            filters=128,
            kernel_size=6,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv3_large",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 4
        self.conv_4_large = tf.layers.conv1d(
            inputs=self.conv_3_large,
            filters=128,
            kernel_size=6,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv4_large",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 2
        self.pool_2_large = tf.layers.max_pooling1d(inputs=self.conv_4_large, pool_size=2, strides=2)

        """
        Frequency Convolutional Layers
        """
        # Conv Layer 1
        self.conv_1_small = tf.layers.conv1d(
            inputs=self.batch_normalizer,
            filters=64,
            kernel_size=self.sampling_rate // 2,
            strides=self.sampling_rate // 16,
            activation=tf.nn.relu,
            padding='SAME',
            kernel_regularizer=self.l2_regulizer,
            name="conv1_small",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 1
        self.pool_1_small = tf.layers.max_pooling1d(inputs=self.conv_1_small, pool_size=8, strides=8)

        # Dropout
        self.dropout_small = tf.layers.dropout(inputs=self.pool_1_small, rate=0.5, training=self.mode == "TRAIN")

        # Conv Layer 2
        self.conv_2_small = tf.layers.conv1d(
            inputs=self.dropout_small,
            filters=128,
            kernel_size=8,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv2_small",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 3
        self.conv_3_small = tf.layers.conv1d(
            inputs=self.conv_2_small,
            filters=128,
            kernel_size=8,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv3_small",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 4
        self.conv_4_small = tf.layers.conv1d(
            inputs=self.conv_3_small,
            filters=128,
            kernel_size=8,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv4_small",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 2
        self.pool_2_small = tf.layers.max_pooling1d(inputs=self.conv_4_small, pool_size=4, strides=4)

        """
        EMG Convolutional Layers
        """
        # Conv Layer 1
        self.conv_1_large_mltch = tf.layers.conv1d(
            inputs=self.batch_normalizer,
            filters=2,
            kernel_size=self.sampling_rate,
            strides=self.sampling_rate,
            activation=tf.nn.relu,
            padding='SAME',
            kernel_regularizer=self.l2_regulizer,
            name="conv1_large_mltch",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 1
        self.pool_1_large_mltch = tf.layers.max_pooling1d(inputs=self.conv_1_large_mltch, pool_size=8, strides=8)

        # Dropout
        self.dropout_large_mltch = tf.layers.dropout(inputs=self.pool_1_large_mltch, rate=0.5,
                                                     training=self.mode == "TRAIN")

        # Conv Layer 2
        self.conv_2_large_mltch = tf.layers.conv1d(
            inputs=self.dropout_large_mltch,
            filters=32,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv2_large_mltch",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 3
        self.conv_3_large_mltch = tf.layers.conv1d(
            inputs=self.conv_2_large_mltch,
            filters=32,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv3_large_mltch",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 4
        self.conv_4_large_mltch = tf.layers.conv1d(
            inputs=self.conv_3_large_mltch,
            filters=128,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv4_large_mltch",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 2
        self.pool_2_large_mltch = tf.layers.max_pooling1d(inputs=self.conv_4_large_mltch, pool_size=2, strides=2)

        """
        EEG/EOG Convolutional Layers
        """
        # Conv Layer 1
        self.conv_1_eeg = tf.layers.conv1d(
            inputs=self.batch_normalizer,
            filters=128,
            kernel_size=self.sampling_rate // 10,
            strides=2,
            activation=tf.nn.relu,
            padding='SAME',
            kernel_regularizer=self.l2_regulizer,
            name="conv1_eeg",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 1
        self.pool_1_eeg = tf.layers.max_pooling1d(inputs=self.conv_1_eeg, pool_size=2, strides=2)

        # Dropout
        self.dropout_eeg = tf.layers.dropout(inputs=self.pool_1_eeg, rate=0.5,
                                                     training=self.mode == "TRAIN")

        # Conv Layer 2
        self.conv_2_eeg = tf.layers.conv1d(
            inputs=self.dropout_eeg,
            filters=128,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv2_eeg",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 3
        self.conv_3_eeg = tf.layers.conv1d(
            inputs=self.conv_2_eeg,
            filters=128,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv3_eeg",
            reuse=tf.AUTO_REUSE
        )

        # Conv Layer 4
        self.conv_4_eeg = tf.layers.conv1d(
            inputs=self.conv_3_eeg,
            filters=128,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv4_eeg",
            reuse=tf.AUTO_REUSE
        )

        # Max Pool Layer 2
        self.pool_2_eeg = tf.layers.max_pooling1d(inputs=self.conv_4_eeg, pool_size=2, strides=2)

        """
        Interchannel Features
        """
        self.extracted_output = tf.concat([self.pool_2_large_mltch, self.pool_2_eeg], axis=1)
        # Conv Layer 1
        self.conv_1_mixed = tf.layers.conv1d(
            inputs=self.extracted_output,
            filters=128,
            kernel_size=6,
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name="conv1_mixed",
            reuse=tf.AUTO_REUSE
        )
        # Max Pool Layer 1
        self.pool_mixed = tf.layers.max_pooling1d(inputs=self.conv_1_mixed, pool_size=4, strides=4)

        """
        CNN Branch Evaluation
        """

        large_output = tf.layers.flatten(self.pool_2_large)
        large_logits = tf.layers.dense(inputs=large_output, units=5, name='large_logits')
        large_correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(large_logits, axis=1))
        self.large_eval = tf.reduce_mean(tf.cast(large_correct_classes, tf.float32))

        small_output = tf.layers.flatten(self.pool_2_small)
        small_logits = tf.layers.dense(inputs=small_output, units=5, name='small_logits')
        small_correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(small_logits, axis=1))
        self.small_eval = tf.reduce_mean(tf.cast(small_correct_classes, tf.float32))

        mltch_output = tf.layers.flatten(self.pool_2_large_mltch)
        mltch_logits = tf.layers.dense(inputs=mltch_output, units=5, name='mltch_logits')
        mltch_correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(mltch_logits, axis=1))
        self.mltch_eval = tf.reduce_mean(tf.cast(mltch_correct_classes, tf.float32))

        eeg_output = tf.layers.flatten(self.pool_2_eeg)
        eeg_logits = tf.layers.dense(inputs=eeg_output, units=5, name='eeg_logits')
        eeg_correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(eeg_logits, axis=1))
        self.eeg_eval = tf.reduce_mean(tf.cast(eeg_correct_classes, tf.float32))

        """
        CNN Output Layer
        """
        # Concatenate all outputs and flatten
        self.cnn_output = tf.concat([self.pool_2_small, self.pool_2_large, self.pool_2_large_mltch, self.pool_2_eeg], axis=1)

        self.dropout = tf.layers.dropout(self.cnn_output, rate=0.5, training=self.mode == "TRAIN")
        self.output_layer = tf.layers.flatten(inputs=self.dropout)

        # Logits Layer, used only in pretraining
        self.logits = tf.layers.dense(inputs=self.output_layer, units=5, reuse=tf.AUTO_REUSE)
        # End Define Model

        """
        Train
        """
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)
        self.optimiser = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learn_rate_pre if self.phase == "PRE" else FLAGS.learn_rate_fine,
            beta1 = 0.9, beta2 = 0.999)
        self.train_op = self.optimiser.minimize(self.loss, global_step=tf.train.get_global_step())

        """
        Eval
        """
        self.correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(self.logits, axis=1))
        self.eval_op = tf.reduce_mean(tf.cast(self.correct_classes, tf.float32))

        """
        Predict
        """
        self.pred_classes = tf.argmax(self.logits, axis=1)

        """
        Save & Restore
        """
        self.saver = tf.train.Saver()  # saves entire model

    def train(self, sess, data):
        self.mode = "TRAIN"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.train_op, self.loss], feed_dict=feed_dict)

    def evaluate(self, sess, data):
        self.mode = "EVAL"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.eval_op], feed_dict=feed_dict)

    def predict(self, sess, data):
        self.mode = "PREDICT"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.pred_classes], feed_dict=feed_dict)

    def checkpoint(self, sess):
        # checkpoint entire model, including separate rep learner
        save_path = self.saver.save(sess, self.rep_learn_dir)
        print("Representation Learner saved to: {}".format(save_path))

    def restore(self, sess):
        print("Restoring Representation Learner...", end=" ")
        if tf.train.checkpoint_exists(self.rep_learn_dir):
            self.saver.restore(sess, self.rep_learn_dir)  # restore only rep learner model
            print("Representation Learner restored.")
        else:
            print("No Representation Learner found at: {}. Initializing.".format(self.rep_learn_dir))
            sess.run(tf.global_variables_initializer())

    def eval_branches(self, sess, data):
        self.mode = "EVAL"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }

        return sess.run([self.large_eval, self.small_eval, self.mltch_eval, self.eeg_eval], feed_dict=feed_dict)