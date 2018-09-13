import tensorflow as tf
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, LSTMCell, DropoutWrapper
from os import path

from src.existing_solution.flags import FLAGS


class SequenceResidualLearner:

    def __init__(self):
        # Housekeeping Parameters
        self.seq_learn_dir = path.join(FLAGS.checkpoint_dir, "seq_learn", "")

        # Hyperparameters
        self.learning_rate = 0.0001
        self.lstm_size = 512
        self.num_lstm_layer = 2

        # Begin Define Model
        """
        Input Layer
        """
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 3072))
        self.y = tf.placeholder(dtype=tf.int32)

        self.input_seqs = tf.reshape(self.x, (FLAGS.sequence_batch_size, FLAGS.sequence_length, 3072))
        self.batch_size = FLAGS.sequence_batch_size

        """
        Bi-Directional LSTM
        """
        # Bidirectional LSTM Cell
        # for some reason, the 2 layers of the bidirectional lstm will not work without explicitly enumerating each layer
        # TODO: make this a loop? or use a different method to create 2 layers
        self.fw_lstm_cell_1 = LSTMCell(num_units=self.lstm_size, use_peepholes=True, state_is_tuple=True)
        self.bw_lstm_cell_1 = LSTMCell(num_units=self.lstm_size, use_peepholes=True, state_is_tuple=True)
        self.fw_lstm_cell_2 = LSTMCell(num_units=self.lstm_size, use_peepholes=True, state_is_tuple=True)
        self.bw_lstm_cell_2 = LSTMCell(num_units=self.lstm_size, use_peepholes=True, state_is_tuple=True)
        # Dropout Between Layers
        self.fw_lstm_cell_1 = DropoutWrapper(self.fw_lstm_cell_1, input_keep_prob=0.5, output_keep_prob=0.5,
                                           state_keep_prob=0.5)
        self.bw_lstm_cell_1 = DropoutWrapper(self.bw_lstm_cell_1, input_keep_prob=0.5, output_keep_prob=0.5,
                                           state_keep_prob=0.5)
        self.fw_lstm_cell_2 = DropoutWrapper(self.fw_lstm_cell_2, input_keep_prob=0.5, output_keep_prob=0.5,
                                             state_keep_prob=0.5)
        self.bw_lstm_cell_2 = DropoutWrapper(self.bw_lstm_cell_2, input_keep_prob=0.5, output_keep_prob=0.5,
                                             state_keep_prob=0.5)

        # 2-Layered Bidirectional LSTM
        self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_lstm_cell_1, self.fw_lstm_cell_2],
                                              state_is_tuple=True)
        self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_lstm_cell_1, self.bw_lstm_cell_2],
                                              state_is_tuple=True)

        self.initial_states_fw = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.initial_states_bw = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)

        # states are dropped after training on each sample so that the states from
        # one sample does not influence those of another
        # dynamic RNN chosen to train on GPUs with smaller memories
        self.bd_lstm, self.states = tf.nn.bidirectional_dynamic_rnn(
            inputs=self.input_seqs,
            cell_fw=self.fw_cell,
            cell_bw=self.bw_cell,
            initial_state_fw=self.initial_states_fw,
            initial_state_bw=self.initial_states_bw
        )

        self.bd_lstm_out = tf.concat(self.bd_lstm, 1)

        """
        Output Layer
        """
        self.batch_normalizer = tf.layers.batch_normalization(
            self.x,
            epsilon=1e-5,
            reuse=tf.AUTO_REUSE,
            name="seq_batch_normalizer"
        )

        self.shortcut_connect = tf.layers.dense(inputs=self.batch_normalizer, units=1024, activation=tf.nn.relu,
                                                name="shorcut_connect")
        #self.lstm_dropout = tf.layers.dropout(inputs=self.bd_lstm, rate=0.5)
        self.output_layer = tf.add(tf.reshape(self.bd_lstm_out, shape=(250, 1024)), self.shortcut_connect)
        self.dropout = tf.layers.dropout(self.output_layer, rate=0.5)
        self.logits = tf.layers.dense(inputs=self.dropout, units=5, name="seq_logits")
        # End Define Model

        """
        Train
        """
        # full model (rep learner + seq rep leaner)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, name="seq_opt")
        self.train_op = self.optimiser.minimize(self.loss, global_step=tf.train.get_global_step(), name="seq_train")
        """
        Eval
        """
        self.correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(self.logits, axis=1))
        self.eval_op = tf.reduce_mean(tf.cast(self.correct_classes, tf.float32), name="seq_eval")

        """
        Predict
        """
        self.pred_classes = tf.argmax(self.logits, axis=1)

        """
        Save & Restore
        """
        self.saver = tf.train.Saver()  # saves only sequence representation learner

    def train(self, sess, rep_learn_output, labels):
        feed_dict = {
            self.x: rep_learn_output,
            self.y: labels
        }
        return sess.run([self.train_op, self.loss], feed_dict=feed_dict)

    def eval(self, sess, rep_learn_output, labels):
        feed_dict = {
            self.x: rep_learn_output,
            self.y: labels
        }
        return sess.run([self.eval_op], feed_dict=feed_dict)

    def predict(self, sess, rep_learn_output):
        feed_dict={
            self.x: rep_learn_output
        }
        return sess.run([self.pred_classes], feed_dict=feed_dict)

    def checkpoint(self, sess):
        # checkpoint entire model, including separate rep learner
        save_path = self.saver.save(sess, self.seq_learn_dir)
        print("Sequential Residual Learner saved to: {}".format(save_path))

    def restore(self, sess):
        if tf.train.checkpoint_exists(self.seq_learn_dir):
            self.saver.restore(sess, self.seq_learn_dir)  # restore only rep learner model
            print("Sequential Residual Learner restored.")
        else:
            raise tf.errors.NotFoundError(
                node_def=self.saver,
                op=self.saver.restore,
                message="No existing Sequential Residual Learner found at: {}".format(self.seq_learn_dir)
            )