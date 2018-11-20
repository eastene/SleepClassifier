import tensorflow as tf

from os import path
from tensorflow.contrib.rnn import stack_bidirectional_rnn, LSTMCell, DropoutWrapper

from src.model.flags import FLAGS
from src.model.RepresentationLearner import RepresentationLearner


class SequenceResidualLearner(RepresentationLearner):

    def __init__(self):
        # Initialize Rep Learner
        super(SequenceResidualLearner, self).__init__()

        # Housekeeping Parameters
        self.seq_learn_dir = path.join(FLAGS.checkpoint_dir, "seq_learn", "")

        # Hyperparameters
        self.learning_rate = 0.0001
        self.lstm_size = 512
        self.num_lstm_layer = 1

        # Begin Define Model
        """
        Input Layer
        """
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 2816))
        self.y = tf.placeholder(dtype=tf.int32)

        self.rep_learn = self.output_layer  # output of representation learner

        self.input_seqs = tf.split(self.rep_learn, num_or_size_splits=25, axis=0)
        self.batch_size = tf.shape(self.input_seqs[0])[0]

        """
        Bi-Directional LSTM
        """
        # Bidirectional LSTM Cell
        self.lstm_cell = LSTMCell(self.lstm_size)

        # Dropout Between Layers
        self.lstm_dropout = DropoutWrapper(self.lstm_cell, input_keep_prob=0.5, output_keep_prob=0.5,
                                           state_keep_prob=0.5)

        # 2-Layered Bidirectional LSTM
        self.initial_states = self.lstm_dropout.zero_state(self.batch_size, dtype=tf.float32)

        # states are dropped after training on each sample so that the states from
        # one sample does not influence those of another
        self.bd_lstm, self.state_fw, self.state_bw = stack_bidirectional_rnn(
            inputs=self.input_seqs,
            cells_fw=[self.lstm_dropout] * self.num_lstm_layer,
            cells_bw=[self.lstm_dropout] * self.num_lstm_layer,
            initial_states_fw=[self.initial_states] * self.num_lstm_layer,
            initial_states_bw=[self.initial_states] * self.num_lstm_layer
        )

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
        self.seq_output_layer = tf.add(tf.reshape(self.bd_lstm, shape=(FLAGS.batch_size, 1024)), self.shortcut_connect)
        self.seq_dropout = tf.layers.dropout(self.seq_output_layer, rate=0.5)
        self.seq_logits = tf.layers.dense(inputs=self.seq_dropout, units=5, name="seq_logits")
        # End Define Model

        """
        Train
        """
        # full model (rep learner + seq rep leaner)
        self.seq_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.seq_logits)
        self.seq_optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    name="seq_opt")
        self.seq_train_op = self.optimiser.minimize(self.seq_loss, global_step=tf.train.get_global_step(),
                                                    name="seq_train")
        """
        Eval
        """
        self.seq_correct_classes = tf.equal(tf.cast(self.y, tf.int64), tf.argmax(self.seq_logits, axis=1))
        self.seq_eval_op = tf.reduce_mean(tf.cast(self.seq_correct_classes, tf.float32), name="seq_eval")

        """
        Predict
        """
        self.seq_pred_classes = tf.argmax(self.seq_logits, axis=1)

        """
        Save & Restore
        """
        self.seq_saver = tf.train.Saver()  # saves only sequence representation learner

    def pretrain(self, sess, data):
        self.mode = "TRAIN"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return super(SequenceResidualLearner, self).pretrain(sess, data)

    def train(self, sess, data):
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.seq_train_op, self.seq_loss], feed_dict=feed_dict)

    def evaluate(self, sess, data):
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.seq_eval_op], feed_dict=feed_dict)

    def evaluate_rep_learner(self, sess, data):
        self.mode = "EVAL"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return super(SequenceResidualLearner, self).evaluate(sess, data)

    def checkpoint(self, sess):
        # checkpoint entire model, including separate rep learner
        super(SequenceResidualLearner, self).checkpoint(sess)
        save_path = self.seq_saver.save(sess, self.seq_learn_dir)
        print("Sequential Learner saved to: {}".format(save_path))

    def restore(self, sess):
        print("Restoring Sequence Learner...", end=" ")
        if tf.train.checkpoint_exists(self.seq_learn_dir):
            self.seq_saver.restore(sess, self.seq_learn_dir)  # restore only rep learner model
            print("Sequential Learner restored.")
        else:
            print("No existing Sequential Learner found at: {}. Initializing.".format(
                self.seq_learn_dir))
            sess.run(tf.global_variables_initializer())
            super(SequenceResidualLearner, self).restore(sess)
