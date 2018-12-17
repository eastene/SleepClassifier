import tensorflow as tf

from os import path
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

from src.model.flags import FLAGS
from src.model.RepresentationLearner import RepresentationLearner


class SequenceResidualLearner(RepresentationLearner):

    def __init__(self):
        # Initialize Rep Learner
        super(SequenceResidualLearner, self).__init__()

        # Housekeeping Parameters
        self.seq_learn_dir = path.join(FLAGS.checkpoint_dir, "seq_learn", "")

        # Hyperparameters
        self.lstm_size = 512
        # self.num_lstm_layer = 2  unused

        # Begin Define Model
        """
        Input Layer
        """
        self.rep_learn = self.output_layer  # output of representation learner
        self.input_seqs = tf.reshape(self.rep_learn, (FLAGS.sequence_batch_size, FLAGS.sequence_length, 3712)) # TODO: change back to 2816?

        # scoped for training with different training rate than representation learner
        with tf.variable_scope("seq_learner") as seq_learner:
            self.seq_batch_size = FLAGS.sequence_batch_size

            """
            Bi-Directional LSTM
            """
            # Bidirectional LSTM Cell
            # for some reason, the 2 layers of the bidirectional lstm will not work
            # without explicitly enumerating each layer
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

            self.states = self.get_state_variables([self.fw_cell, self.bw_cell])

            # self.initial_states_fw = self.fw_cell.zero_state(self.seq_batch_size, dtype=tf.float32)
            # self.initial_states_bw = self.bw_cell.zero_state(self.seq_batch_size, dtype=tf.float32)

            # states are dropped after training on each sample so that the states from
            # one sample does not influence those of another
            # dynamic RNN chosen to train on GPUs with smaller memories
            self.bd_lstm, self.new_states = tf.nn.bidirectional_dynamic_rnn(
                inputs=self.input_seqs,
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                initial_state_fw=self.states[0],
                initial_state_bw=self.states[1]
            )

            self.update_op = self.get_state_update_op(self.states, self.new_states)

            # concatenate forward and backward lstm outputs
            self.bd_lstm_out = tf.concat(self.bd_lstm, 1)

            """
            Output Layer
            """
            self.seq_batch_normalizer = tf.layers.batch_normalization(
                self.rep_learn,
                epsilon=1e-5,
                reuse=tf.AUTO_REUSE,
                name="seq_batch_normalizer"
            )

            self.shortcut_connect = tf.layers.dense(inputs=self.seq_batch_normalizer, units=1024, activation=tf.nn.relu,
                                                    name="shorcut_connect")
            # self.lstm_dropout = tf.layers.dropout(inputs=self.bd_lstm, rate=0.5)
            self.seq_output_layer = tf.add(
                tf.reshape(self.bd_lstm_out, shape=(FLAGS.sequence_length * self.seq_batch_size, 1024)),
                self.shortcut_connect)
            self.seq_dropout = tf.layers.dropout(self.seq_output_layer, rate=0.5)
            self.seq_logits = tf.layers.dense(inputs=self.seq_dropout, units=5, name="seq_logits")
            # End Define Model

            """
            Train
            """
            # full model (rep learner + seqs rep leaner)
            self.seq_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.seq_logits)
            self.seq_optimiser = tf.train.AdamOptimizer(learning_rate=FLAGS.learn_rate_pre, beta1=0.9, beta2=0.999,
                                                        name="adam_seq")
            # TODO: make sure this is only training parts it is meant to train
            self.gvs = self.seq_optimiser.compute_gradients(self.seq_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                       scope='seq_learner'))
            self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if var == "seq_learner/bidirectional_rnn"
                               else (grad, var) for grad, var in self.gvs]

            self.seq_train_op = self.seq_optimiser.apply_gradients(self.capped_gvs)
            #self.seq_train_op = self.seq_optimiser.minimize(self.seq_loss, global_step=tf.train.get_global_step(),
            #                                                name="seq_train",
            #                                                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            #                                                                           scope='seq_learner'))

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

    def get_state_variables(self, cells):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        states = []
        for cell in cells:
            state_variables = []
            for state_c, state_h in cell.zero_state(self.seq_batch_size, tf.float32):
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
            # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
            states.append(tuple(state_variables))
        return states

    def get_state_update_op(self, cell_states, new_cell_states):
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variables, new_states in zip(cell_states, new_cell_states):
            for state_variable, new_state in zip(state_variables, new_states):
                # Assign the new state to the state variables on this layer
                update_ops.extend([state_variable[0].assign(new_state[0]),
                                   state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

    def pretrain(self, sess, data):
        return super(SequenceResidualLearner, self).train(sess, data)

    def reset_lstm_state(self, sess):
        zero_states = [
            self.fw_cell.zero_state(self.seq_batch_size, tf.float32),
            self.bw_cell.zero_state(self.seq_batch_size, tf.float32)
        ]
        return sess.run(self.get_state_update_op(self.states, zero_states))

    def train(self, sess, data):
        self.mode = "TRAIN"
        self.learning_rate = FLAGS.learn_rate_fine  # turn down the learning rate for the feature learner
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.seq_train_op, self.train_op_fine, self.seq_loss, self.loss, self.update_op], feed_dict=feed_dict)

    def evaluate_rep_learner(self, sess, data):
        return super(SequenceResidualLearner, self).evaluate(sess, data)

    def evaluate(self, sess, data):
        self.mode = "EVAL"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.seq_eval_op, self.update_op], feed_dict=feed_dict)

    def checkpoint(self, sess):
        # checkpoint entire model, including separate rep learner
        super(SequenceResidualLearner, self).checkpoint(sess)
        save_path = self.seq_saver.save(sess, self.seq_learn_dir)
        print("Sequential Learner saved to: {}".format(save_path))

    def predict(self, sess, data):
        self.mode = "PREDICT"
        feed_dict = {
            self.x: data[0],
            self.y: data[1]
        }
        return sess.run([self.seq_pred_classes], feed_dict=feed_dict)

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
