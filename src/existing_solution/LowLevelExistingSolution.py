import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix

from src.existing_solution.flags import FLAGS, FILE_PATTERN


def pretrain():
    from src.existing_solution.input_pipeline import InputPipeline
    from src.existing_solution.representation_learner import RepresentationLearner
    from src.existing_solution.sequence_residual_learner import SequenceResidualLearner
    from src.existing_solution.sequential_input_pipeline import SequentialInputPipeline

    # hyper-parameters
    n_folds = 20
    sampling_rate = 100

    # model
    rep_learn = RepresentationLearner(sampling_rate)
    seq_learn = SequenceResidualLearner()

    # batch iterator
    ip = InputPipeline(FILE_PATTERN, size_of_split=10)
    next_elem_train = ip.next_train_elem()
    next_elem_eval = ip.next_eval_elem()

    # define initialisation operator
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # initialize or restore
        try:
            rep_learn.restore(sess)
        except tf.errors.NotFoundError:
            sess.run(init_op)

        # PRETRAINING TRAIN LOOP
        for epoch in range(FLAGS.num_epochs_pretrain):
            sess.run(ip.initialize_train())
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
        sess.run(ip.initialize_eval())

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

        # Sequence learning input
        seq_in_train = SequentialInputPipeline("/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_data/SC40*.npz")
        seq_in_eval = SequentialInputPipeline(
            "/home/evan/PycharmProjects/SleepClassifier/data/existing_solution/prepared_data/SC41*.npz")

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        # Finetuning
        with tf.Session() as sess:
            # initialize or restore
            try:
                seq_learn.restore(sess)
            except tf.errors.NotFoundError:
                sess.run(init_op)

            try:
                rep_learn.restore(sess)
            except tf.errors.NotFoundError:
                sess.run(init_op)

            # FINETUNING TRAIN LOOP
            for epoch in range(FLAGS.num_epochs_pretrain):
                seq_in_train.reinitialize()
                cost = 0.0
                n_batches = 0

                # each epoch consits of a number of patient sequences
                for seq_data in seq_in_train:
                    # each patient sequence is batched, and the LSTM is reinitialized for each patient
                    for x, y in seq_data:
                        _, rep_out = rep_learn.finetune(sess, [x, y])
                        _, c = seq_learn.train(sess, rep_out, y)
                        cost += c
                        n_batches += 1

                if epoch % 3 == 0:
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost / n_batches))
                    rep_learn.checkpoint(sess)
                    seq_learn.checkpoint(sess)

            # FINETUNING EVAL LOOP
            m_tot = 0
            n_batches = 0

            # Evaluate
            # each epoch consits of a number of patient sequences
            for seq_data in seq_in_eval:
                # each patient sequence is batched, and the LSTM is reinitialized for each patient
                for x, y in seq_data:
                    rep_out = rep_learn.eval_finetune(sess, [x, y])
                    m = seq_learn.eval(sess, np.squeeze(rep_out), y)
                    n_batches += 1
                    m_tot += m[0]

            print("Finetuning Accuracy: {}".format(m_tot / n_batches))

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


def main(unused_argv):
    pretrain()
    #fine_tune(tf.estimator.ModeKeys.TRAIN)


if __name__ == "__main__":
    tf.app.run()