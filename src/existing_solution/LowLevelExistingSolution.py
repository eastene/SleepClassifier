import tensorflow as tf
import numpy as np
import glob

from tensorflow.contrib.layers import l2_regularizer
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler

from src.existing_solution.flags import FLAGS, TRAIN_FILE_PATTERN, EVAL_FILE_PATTERN


def pretrain():
    from src.existing_solution.input_pipeline import InputPipeline
    from src.existing_solution.representation_learner import RepresentationLearner
    from src.existing_solution.sequence_residual_learner import SequenceResidualLearner

    # hyper-parameters
    n_folds = 20
    sampling_rate = 100

    # model
    rep_learn = RepresentationLearner(sampling_rate)
    seq_learn = SequenceResidualLearner()

    # train batch iterator
    ip_train = InputPipeline(TRAIN_FILE_PATTERN)
    next_elem_train = ip_train.next_elem()
    # eval batch iterator
    ip_eval = InputPipeline(EVAL_FILE_PATTERN)
    next_elem_eval = ip_eval.next_elem()

    # define initialisation operator
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # initialize or restore
        try:
            rep_learn.restore_pretrainer(sess)
        except tf.errors.NotFoundError:
            sess.run(init_op)

        # PRETRAINING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            sess.run(ip_train.initializer())
            cost = 0.0
            n_batches = 0

            try:
                while True:
                    data = sess.run(next_elem_train)
                    _, c = rep_learn.pretrain(sess, data)
                    cost += c
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass  # reached end of epoch

            if epoch % 3 == 0:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                rep_learn.checkpoint(sess)

        # PRETRAINING EVAL LOOP
        m_tot = 0
        n_batches = 0
        sess.run(ip_eval.initializer())

        # Evaluate
        try:
            while True:
                data = sess.run(next_elem_eval)
                m = rep_learn.eval(sess, data)
                n_batches += 1
                m_tot += m[0]

        except tf.errors.OutOfRangeError:
            pass  # reached end of epoch

        print("Pretraining Accuracy: {}".format(m_tot / n_batches))


        # Finetuning
        with tf.Session() as sess:
            # initialize or restore
            try:
                seq_learn.restore(sess)
            except tf.errors.NotFoundError:
                sess.run(init_op)

            try:
                rep_learn.restore_representation_learner(sess)
            except tf.errors.NotFoundError:
                sess.run(init_op)

            # PRETRAINING TRAIN LOOP
            for epoch in range(FLAGS.num_epochs_pretrain):
                sess.run(ip_train.initializer())
                cost = 0.0
                n_batches = 0

                try:
                    while True:
                        data = sess.run(ip_train.next_elem())
                        _, rep_out, labels = rep_learn.finetune(sess, data)
                        _, c = seq_learn.train(sess, rep_out, labels)
                        cost += c
                        n_batches += 1
                except tf.errors.OutOfRangeError:
                    pass  # reached end of epoch

                if epoch % 3 == 0:
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                    rep_learn.checkpoint_representation_learner(sess)

"""
        # Test after all epochs
        predictions = sess.run(pred_classes, {x:test_data, y:test_labels})
        con_mat = tf.confusion_matrix(labels=list(test_labels), predictions=list(predictions))
        print("Results of Pretraining Feature Representaion:")
        print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))
"""


def main(unused_argv):
    pretrain()
    #fine_tune(tf.estimator.ModeKeys.TRAIN)


if __name__ == "__main__":
    tf.app.run()