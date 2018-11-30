import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from src.model.flags import FLAGS
from src.model.InputPipeline import InputPipeline
from src.model.RepresentationLearner import RepresentationLearner
from src.model.SequenceResidualLearner import SequenceResidualLearner


class DeepSleepNet:

    def __init__(self):
        # hyper-parameters
        self.n_folds = 20
        self.sampling_rate = FLAGS.sampling_rate

        # model
        self.seq_learn = SequenceResidualLearner()


        # batch iterator
        self.input = InputPipeline()
        self.next_elem_train_pre = self.input.next_train_elem()
        self.next_elem_eval_pre = self.input.next_eval_elem()
        self.next_elem_train_fine = self.input.next_train_elem
        self.next_elem_eval_fine = self.input.next_eval_elem

        # define initialisation operator
        self.init_op = tf.global_variables_initializer()

        # loss arrays for plotting loss over time
        self.loss_tr_pre = np.empty((FLAGS.num_epochs_pretrain))
        self.loss_ts_pre = np.empty((FLAGS.num_epochs_pretrain))
        self.loss_tr_fine = np.empty((FLAGS.num_epochs_finetune))
        self.loss_ts_fine = np.empty((FLAGS.num_epochs_finetune))

    def run_epoch_pretrain(self, sess):
        # PRETRAINING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            sess.run(self.input.initialize_train())
            cost = 0.0
            n_batches = 0

            try:
                while True:
                    data = sess.run(self.next_elem_train_pre)
                    _, c = self.seq_learn.pretrain(sess, data)
                    cost += c
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                self.seq_learn.checkpoint(sess)

    def run_epoch_finetune(self, sess):
        # FINETUNING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            self.input.initialize_train()
            cost = 0.0
            n_batches = 0

            # each epoch consists of a number of patient sequences
            try:
                while True:
                    seq_data = self.next_elem_train_fine(sequential=True)
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    for batch in seq_data:
                        _, c = self.seq_learn.train(sess, batch)
                        cost += c
                        n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                self.seq_learn.checkpoint(sess)

    def train(self):
        with tf.Session() as sess:
            """
            Train Representation Learner (Pretraining)
            """
            # initialize or restore
            self.seq_learn.restore(sess)

            print("Pretraining for {} Epochs.".format(FLAGS.num_epochs_pretrain))
            self.run_epoch_pretrain(sess)
            print("Evaluating Representation Learner...", end=" ")
            self.evaluate(sess, rep_only=True)

            """
            Train Sequence Learner (Finetuning)
            """
            self.run_epoch_finetune(sess)
            print("Evaluating Model...", end=" ")
            self.evaluate()

            if FLAGS.confsn_mat:
                self.print_confusion_matrix(sess)
            if FLAGS.plot_loss:
                self.plot_loss()

    def evaluate(self, sess=None, rep_only=False):
        if rep_only:
            # PRETRAINING EVAL LOOP
            m_tot = 0
            n_batches = 0
            sess.run(self.input.initialize_eval())

            # Evaluate
            try:
                while True:
                    data = sess.run(self.next_elem_eval_pre)
                    m = self.seq_learn.evaluate_rep_learner(sess, data)
                    n_batches += 1
                    m_tot += m[0]

            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            print("Representation Learner Accuracy: {}".format(m_tot / n_batches))

        else:
            # FINETUNING EVAL LOOP
            m_tot = 0
            n_batches = 0
            self.input.initialize_train()

            # Evaluate
            try:
                while True:
                    seq_data = self.next_elem_eval_fine(sequential=True)
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    for batch in seq_data:
                        m = self.seq_learn.evaluate(sess, batch)
                        n_batches += 1
                        m_tot += m[0]

            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            print("Overall Accuracy: {}".format(m_tot / n_batches))

    def predict(self):
        pass

    def print_confusion_matrix(self, sess):
        # Print final Confusion Matrix
        Y = []
        Y_pred = []
        self.input.initialize_eval()
        try:
            while True:
                seq_data = self.next_elem_train_fine
                # each patient sequence is batched, and the LSTM is reinitialized for each patient
                for batch in seq_data:
                    y_pred = self.seq_learn.predict(sess, batch)
                    Y.append(batch[1].flatten().tolist())
                    Y_pred.append(np.vstack(y_pred).flatten().tolist())
        except tf.errors.OutOfRangeError:
            pass  # reached end of epoch

        labels = [l for arr in Y for l in arr]
        predictions = [l for arr in Y_pred for l in arr]
        print("\n\nConfusion Matrix:")
        print(confusion_matrix(labels, predictions))

    def plot_loss(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(range(1, FLAGS.num_epochs_pretrain), self.loss_tr_pre)
        ax1.plot(range(1, FLAGS.num_epochs_pretrain), self.loss_ts_pre)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Cross Entropy)')
        ax1.title('Representation Learning')
        ax2.plot(range(1, FLAGS.num_epochs_finetune), self.loss_tr_fine)
        ax2.plot(range(1, FLAGS.num_epochs_finetune), self.loss_ts_fine)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (Cross Entropy)')
        ax2.title('Sequence Residual Learning')


def main(unused_argv):
    dn = DeepSleepNet()
    dn.train()
    dn.plot_loss()
    # fine_tune(tf.estimator.ModeKeys.TRAIN)


if __name__ == "__main__":
    tf.app.run()
