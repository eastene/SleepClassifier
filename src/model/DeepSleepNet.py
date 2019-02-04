import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import environ
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from src.model.InputPipeline import InputPipeline
from src.model.SequenceResidualLearner import SequenceResidualLearner
from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE


class DeepSleepNet:

    def __init__(self):
        # set GPU parameters
        environ["CUDA_VISIBLE_DEVICES"] = "1"

        # hyper-parameters
        self.sampling_rate = EFFECTIVE_SAMPLE_RATE

        # model
        self.seq_learn = SequenceResidualLearner()

        # batch iterator
        self.input = InputPipeline()

        # define initialisation operator
        self.init_op = tf.global_variables_initializer()

        # loss arrays for plotting loss over time
        self.loss_tr_pre = np.empty(FLAGS.num_epochs_pretrain)
        self.loss_tr_fine = np.empty(FLAGS.num_epochs_finetune)
        self.loss_tr_fine_seq = np.empty(FLAGS.num_epochs_finetune)

    def run_epoch_pretrain(self, sess):
        # PRETRAINING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            self.input.initialize_train()
            cost = 0.0
            n_batches = 0

            try:
                while True:
                    data = self.input.next_train_elem()
                    _, c = self.seq_learn.pretrain(sess, data)
                    cost += c
                    n_batches += 1
            except IndexError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                self.seq_learn.checkpoint(sess)

            self.loss_tr_pre[epoch] = cost / n_batches

    def run_epoch_finetune(self, sess):
        # FINETUNING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_finetune):
            self.input.initialize_train(sequential=True)
            cost_seq = 0.0
            cost = 0.0
            n_batches = 0

            # each epoch consists of a number of patient sequences
            try:
                while True:
                    seq_data = self.input.next_train_elem(sequential=True)
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    self.seq_learn.reset_lstm_state(sess)
                    for batch in seq_data:
                        c, c_seq = self.seq_learn.train(sess, batch)
                        cost_seq += c_seq
                        cost += c
                        n_batches += 1
            except IndexError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:",
                      (epoch + 1),
                      "Representation cost = {:.3f}  Sequence cost = {:.3f}".format(
                        cost / n_batches, cost_seq / n_batches))
                self.seq_learn.checkpoint(sess)

            self.loss_tr_fine[epoch] = cost / n_batches
            self.loss_tr_fine_seq[epoch] = cost_seq / n_batches

    def train(self):
        with tf.Session() as sess:
            """
            Train Representation Learner (Pretraining)
            """
            sess.run(self.init_op)
            # initialize or restore
            self.seq_learn.restore(sess)
            print("Pretraining for {} Epochs.".format(FLAGS.num_epochs_pretrain))
            self.run_epoch_pretrain(sess)
            print("Evaluating Representation Learner...", end=" ")
            self.evaluate(sess, rep_only=True)

            #train_writer = tf.summary.FileWriter('train', sess.graph)
            #train_writer.add_graph(tf.get_default_graph())

            if FLAGS.cnfsn_mat:
                print("Pretraining Performance:")
                self.print_confusion_matrix(sess, rep_only=True)

            """
            Train Sequence Learner (Finetuning)
            """
            print("Finetuning for {} Epochs.".format(FLAGS.num_epochs_finetune))
            self.run_epoch_finetune(sess)
            print("Evaluating Model...", end=" ")
            self.evaluate(sess)

            if FLAGS.cnfsn_mat:
                print("Model Performance:")
                self.print_confusion_matrix(sess)

            if FLAGS.plot_loss:
                self.plot_loss()

    def evaluate(self, sess=None, rep_only=False):
        if rep_only:
            # PRETRAINING EVAL LOOP
            m_tot = 0
            n_batches = 0
            self.input.initialize_eval()

            # Evaluate
            try:
                while True:
                    data = self.input.next_eval_elem()
                    m = self.seq_learn.evaluate_rep_learner(sess, data)
                    n_batches += 1
                    m_tot += m[0]

            except IndexError:
                pass  # reached end of epoch

            print("Representation Learner Accuracy: {}".format(m_tot / n_batches))
        else:
            # FINETUNING EVAL LOOP
            m_tot = 0
            n_batches = 0
            self.input.initialize_eval(sequential=True)

            # Evaluate
            try:
                while True:
                    seq_data = self.input.next_eval_elem(sequential=True)
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    self.seq_learn.reset_lstm_state(sess)
                    for batch in seq_data:
                        m, _ = self.seq_learn.evaluate(sess, batch)
                        n_batches += 1
                        m_tot += m

            except IndexError:
                pass  # reached end of epoch

            print("Overall Accuracy: {}".format(m_tot / n_batches))

    def print_confusion_matrix(self, sess, rep_only=False):
        # Print final Confusion Matrix
        Y = []
        Y_pred = []
        if rep_only:
            self.input.initialize_eval()
            try:
                while True:
                    data = self.input.next_eval_elem()
                    y_pred = self.seq_learn.predict_rep_learner(sess, data)
                    Y.append(data[1].astype(np.int64))
                    Y_pred.append(np.vstack(y_pred).flatten())
            except IndexError:
                pass  # reached end of epoch

        else:
            self.input.initialize_eval(sequential=True)
            try:
                while True:
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    data = self.input.next_eval_elem(sequential=True)
                    for batch in data:
                        self.seq_learn.reset_lstm_state(sess)
                        y_pred = self.seq_learn.predict(sess, batch)
                        Y.append(batch[1].flatten())
                        Y_pred.append(np.vstack(y_pred).flatten())
            except IndexError:
                pass  # reached end of epoch

        labels = np.hstack(Y)
        predictions = np.hstack(Y_pred)
        self.print_performance(labels, predictions)

    @staticmethod
    def print_performance(labels, predictions):
        print("\n\nConfusion Matrix:")
        print(confusion_matrix(labels, predictions))

        # Print recall, precision, F1, and support
        stages = ["N1", "N2", "N3", "REM", "Wake"]
        p, r, f, s = precision_recall_fscore_support(labels, predictions)
        for i, stage in enumerate(stages):
            print(stage, end=" ")
            print("Precision = {:1.3f}, Recall = {:1.3f}, F1 = {:1.3f}, Support = {}".format(p[i], r[i], f[i], s[i]))

    def plot_loss(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(range(1, FLAGS.num_epochs_pretrain + 1), self.loss_tr_pre)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Cross Entropy)')
        ax1.set_title('Representation Learning')
        ax2.plot(range(1, FLAGS.num_epochs_finetune + 1), self.loss_tr_fine)
        ax2.plot(range(1, FLAGS.num_epochs_finetune + 1), self.loss_tr_fine_seq)
        ax2.legend(['Representation', 'Sequence'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (Cross Entropy)')
        ax2.set_title('Sequence Residual Learning')
        plt.show()


def main(unused_argv):
    dn = DeepSleepNet()
    dn.train()


if __name__ == "__main__":
    tf.app.run()
