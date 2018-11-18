import tensorflow as tf

from src.model.flags import FLAGS
from src.model.InputPipeline import InputPipeline
from src.model.RepresentationLearner import RepresentationLearner
from src.model.sequence_residual_learner import SequenceResidualLearner

class DeepSleepNet:

    def __init__(self):
        # hyper-parameters
        self.n_folds = 20
        self.sampling_rate = FLAGS.sampling_rate

        # model
        self.rep_learn = RepresentationLearner()
        self.seq_learn = SequenceResidualLearner()

        # batch iterator
        self.input = InputPipeline()
        self.next_elem_train_pre = self.input.next_train_elem()
        self.next_elem_eval_pre = self.input.next_eval_elem()
        self.next_elem_train_fine = self.input.next_train_elem(sequential=True)
        self.next_elem_eval_fine = self.input.next_eval_elem(sequential=True)

        # define initialisation operator
        self.init_op = tf.global_variables_initializer()

    def run_epoch_pretrain(self, sess):
        # PRETRAINING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            sess.run(self.input.initialize_train())
            cost = 0.0
            n_batches = 0

            try:
                while True:
                    data = sess.run(self.next_elem_train_pre)
                    _, c = self.rep_learn.pretrain(sess, data)
                    cost += c
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                self.rep_learn.checkpoint(sess)

    def run_epoch_finetune(self, sess):
        # FINETUNING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            self.input.initialize_train()
            cost = 0.0
            n_batches = 0

            # each epoch consits of a number of patient sequences
            try:
                while True:
                    seq_data = self.next_elem_train_fine
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    for batch in seq_data:
                        _, c = self.seq_learn.train(sess, batch)
                        cost += c
                        n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                self.rep_learn.checkpoint(sess)
                self.seq_learn.checkpoint(sess)

    def train(self):
        with tf.Session() as sess:
            """
            Train Representation Learner (Pretraining)
            """
            # initialize or restore
            try:
                self.rep_learn.restore(sess)
            except tf.errors.NotFoundError:
                sess.run(self.init_op)

            print("Pretraining for {} Epochs.".format(FLAGS.num_epochs_pretrain))
            self.run_epoch_pretrain(sess)
            print("Evaluating Representation Learner...", end=" ")
            self.evaluate(sess, rep_only=True)

            """
            Train Sequence Learner (Finetuning)
            """
            # initialize or restore
            try:
                self.seq_learn.restore(sess)
            except tf.errors.NotFoundError:
                sess.run(self.init_op)
                self.rep_learn.restore(sess)  # restore as this will be reinitialized by above step

            self.run_epoch_finetune(sess)
            print("Evaluating Model...", end=" ")
            self.evaluate()

            # Print final Confusion Matrix
            Y = []
            Y_pred = []
            seq_in_eval.reinitialize()
            for seq_data in seq_in_eval:
                # each patient sequence is batched, and the LSTM is reinitialized for each patient
                for x, y in seq_data:
                    rep_out = rep_learn.eval_finetune(sess, [x, y])
                    y_pred = seq_learn.predict(sess, np.squeeze(rep_out))
                    Y.append(y.flatten().tolist())
                    Y_pred.append(np.vstack(y_pred).flatten().tolist())

            labels = [l for arr in Y for l in arr]
            predictions = [l for arr in Y_pred for l in arr]
            print("\n\nConfusion Matrix:")
            print(confusion_matrix(labels, predictions))

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
                    m = self.rep_learn.eval(sess, data)
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
                    seq_data = self.next_elem_eval_fine
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    for batch in seq_data:
                        m = self.seq_learn.eval(sess, batch)
                        n_batches += 1
                        m_tot += m[0]

            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            print("Overall Accuracy: {}".format(m_tot / n_batches))

    def predict(self):
        pass


def main(unused_argv):
    dn = DeepSleepNet()
    dn.train()
    #fine_tune(tf.estimator.ModeKeys.TRAIN)


if __name__ == "__main__":
    tf.app.run()
