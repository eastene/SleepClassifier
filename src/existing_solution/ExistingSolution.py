import glob

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import stack_bidirectional_rnn, LSTMCell, DropoutWrapper
from tensorflow.contrib.layers import l2_regularizer
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler
tf.logging.set_verbosity(tf.logging.INFO)


def representation_learner(data, sampling_rate, mode, use_small_filter=True, trainable=True):
    """
    CNN with small filter size
    :param data: EEG signal input
    :param labels: Sleep stage labels
    :param sampling_rate: rate at which the EEG input is sampled
    :param mode: tensorflow mode (see tf.estimator.ModeKeys)
    :param use_small_filter: if true, sets filter size to sampling_rate / 2, else sets to sampling_rate * 4
    :return:
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
    batch_normalizer = tf.layers.batch_normalization(
        input_layer,
        epsilon=1e-5,
        reuse=tf.AUTO_REUSE,
        name="conv_batch_normalizer"
    )

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
        reuse=tf.AUTO_REUSE,
        trainable=trainable
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
        reuse=tf.AUTO_REUSE,
        trainable=trainable
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
        reuse=tf.AUTO_REUSE,
        trainable=trainable
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
        reuse=tf.AUTO_REUSE,
        trainable=trainable
    )

    # Max Pool Layer 2
    pool_2 = tf.layers.max_pooling1d(inputs=conv_4, pool_size=max_pool_size // 2, strides=pool_stride // 2)

    return pool_2


def sequence_residual_learner_fn(features, labels, mode, params):
    """
    Train the entire DeepSleepNet model using the parameters learned in the pretraining phase.
    :param features: 30s of EEG signal sampled at sampling_rate times per second
    :param labels: sleep stage labels (1-5)
    :param mode: training/eval/predict mode for the model
    :param params: dict containing "learn_rate" (Learning Rate), "fs" (EEG Sampling Rate - 100hz)
    :return: None
    """

    # retrieve hyperparameters
    learn_rate = params['learn_rate']
    sampling_rate = params['fs']

    # Input Layer (batch_size * feature_size)
    input_layer = tf.layers.flatten(features['x'])

    with tf.variable_scope("rep_scope"):

        # CNN Portion (small and large filters)
        small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True, trainable=False)
        large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False, trainable=False)

        # Concatenate results of both CNNs
        cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
        cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten tensor of shape (batch_size, out_width, out_channels) to (batch_size, out_width * out_channels)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)

    with tf.variable_scope("seq_scope"):
        lstm_size = 512
        input_seqs = tf.split(flat_layer, num_or_size_splits=25, axis=0)
        batch_size = tf.shape(input_seqs[0])[0]

        # Bidirectional LSTM Cell
        lstm_cell = LSTMCell(lstm_size)

        # Dropout Between Layers
        lstm_dropout = DropoutWrapper(lstm_cell, input_keep_prob=0.5, output_keep_prob=0.5, state_keep_prob=0.5)

        # 2-Layered Bidirectional LSTM
        initial_states = lstm_dropout.zero_state(batch_size, dtype=tf.float32)

        num_layer = 1
        # states are dropped after training on each sample so that the states from
        # one sample does not influence those of another
        output, state_fw, state_bw = stack_bidirectional_rnn(
            inputs=input_seqs,
            cells_fw=[lstm_dropout] * num_layer,
            cells_bw=[lstm_dropout] * num_layer,
            initial_states_fw=[initial_states] * num_layer,
            initial_states_bw=[initial_states] * num_layer
        )

        batch_normalizer = tf.layers.batch_normalization(
            flat_layer,
            epsilon=1e-5,
            reuse=tf.AUTO_REUSE,
            name="seq_batch_normalizer"
        )

        shortcut_connect = tf.layers.dense(inputs=batch_normalizer, units=1024, activation=tf.nn.relu, name="shorcut_connect")
        concat_layer = tf.add(tf.reshape(output, shape=(250, 1024)), shortcut_connect)
        concat_dropout = tf.layers.dropout(concat_layer, rate=0.5)
        logits = tf.layers.dense(inputs=concat_dropout, units=5, name="seq_logits")


    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'predictions': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def representation_learner_fn(features, labels, mode, params):
    """
    Pretrain the feature representation learning model (CNNs with large and small filters) only (No LSTM)
    Expects to be trained on a dataset with some sort of supersampling of underrepresented classes (e.g. N1)
    in order to train the model's feature representation learning with features that represent all classes well.
    :param features: EEG signals over 30s intervals for each batch
    :param labels: Sleep stage labels (W, N1, N2, N3, REM) for each batch
    :param mode: Mode to run function in [Training, Eval, Test] (see tf.estimator.ModeKeys)
    :param params: dict containing "learn_rate" (Learning Rate), "fs" (EEG Sampling Rate - 100hz)
    :return: EstimatorSpec
    """

    # retrieve hyperparameters
    learn_rate = params['learn_rate']
    sampling_rate = params['fs']

    # Input Layer (batch_size * feature_size)
    input_layer = features['x']

    with tf.variable_scope("rep_scope"):
        # CNN Portion (small and large filters)
        small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True)
        large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False)

        # Concatenate results of both CNNs
        cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
        cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Flatten tensor of shape (batch_size, out_width, out_channels) to (batch_size, out_width * out_channels)
        flat_layer = tf.layers.flatten(inputs=cnn_dropout)

        # Softmax layer only used in pretraining, the weights to this layer are dropped after training
        # Size is now (batch_size * 5)
        logits = tf.layers.dense(inputs=flat_layer, units=5, name='logits')

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'predictions': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def load_data():
    train_files = glob.glob('../../data/existing_solution/prepared_data/SC4*.npz')
    X = []
    Y = []
    sampling_rate = 0

    for f in train_files:
        data = np.load(f)
        X.append(data['x'])
        Y.append(data['y'])
        sampling_rate = data['fs']

    return np.vstack(X), np.hstack(Y), sampling_rate


def load_seq_data():
    train_files = glob.glob('../../data/existing_solution/prepared_data/SC4*.npz')
    X = []
    Y = []
    sampling_rate = 0

    for f in train_files:
        data = np.load(f)
        X.append(np.squeeze(data['x']))
        Y.append(data['y'])
        sampling_rate = data['fs']

    return X, Y, sampling_rate


def make_fold_seq(data, labels, k, fold_size, num_examples):
    train_fold_start = k * fold_size
    train_fold_end = ((k + 19) * fold_size) % num_examples
    eval_fold_end = ((k + 20) * fold_size) % num_examples

    train_data = data[train_fold_start: train_fold_end]
    train_labels = labels[train_fold_start: train_fold_end]
    eval_data = data[train_fold_end: eval_fold_end]
    eval_labels = labels[train_fold_end: eval_fold_end]

    return train_data, train_labels, eval_data, eval_labels


def make_seq_data(data, labels, batch_size, seq_len):
    batch_len = data.shape[0] // batch_size
    epoch_len = batch_len // seq_len

    # (batch_size, batch_len, 3000, 1)
    seq_data = np.zeros((batch_size, batch_len) + data.shape[1:], dtype=data.dtype)
    # (batch_size, batch_len, 1, 1)
    seq_labels = np.zeros((batch_size, batch_len) + labels.shape[1:], dtype=labels.dtype)

    for i in range(batch_size):
        seq_data[i] = data[i * batch_len: (i+1) * batch_len]
        seq_labels[i] = labels[i * batch_len: (i + 1) * batch_len]

    X = []
    Y = []
    for i in range(epoch_len):
        X.append(seq_data[:, i * seq_len: (i+1) * seq_len].reshape((-1, ) + data.shape[1:]))
        Y.append(seq_labels[:, i * seq_len: (i + 1) * seq_len].reshape((-1, ) + labels.shape[1:]))

    return X, Y


def main(argv):

    # TODO: restore checkpointing
    # turn off checkpointing
    no_chkpt_run_config = tf.estimator.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=None)

    # ******** Pretrain Representation Learning Model ********

    # Load dataset
    data, labels, sampling_rate = load_data()
    x = data.shape[0]
    y = data.shape[1]

    # Set aside some data to test model after training
    fold_data, test_data, fold_labels, test_labels = train_test_split(np.reshape(data, (x, y)), labels, test_size=0.1)

    # Oversample data so that each class has the same number of subjects
    print("Pre Oversampling Label Counts {}".format(np.bincount(labels)))
    rus = RandomOverSampler()
    os_data, os_labels = rus.fit_sample(fold_data, y=fold_labels)
    os_data = os_data.astype(np.float32)  # SMOTE outputs float64, which causes issues with Tensorflow
    print("Post Oversampling Label Counts {}".format(np.bincount(os_labels)))

    # Pretrainer Hyperparameters
    pretrainer_params = {
        "learn_rate": 0.0001,
        "fs": sampling_rate
    }

    # Create Pretrain Estimator
    pretrainer = tf.estimator.Estimator(
        model_fn=representation_learner_fn,
        model_dir="/tmp/rep_learn_model",
        params=pretrainer_params,
        config=no_chkpt_run_config
    )
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Perform k=20 cross validation
    folds = KFold(n_splits=20, shuffle=False)

    """
    # Loop over Splits
    k = 0
    for train_idx, res_idx in folds.split(os_data):

        # Define Fold
        train_data = os_data[train_idx]
        train_labels = os_labels[train_idx]
        eval_data = os_data[res_idx]
        eval_labels = os_labels[res_idx]

        # Generate Training Function
        train_pretain_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=100,
            shuffle=False
        )

        # Train the model
        pretrainer.train(
            input_fn=train_pretain_fn,
            steps=100,
            hooks=[logging_hook]
        )

        # Evaluate Model
        eval_pretrain_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        # Print Results of Evaluation
        eval_results = pretrainer.evaluate(input_fn=eval_pretrain_fn)
        print("Results of fold {}:".format(k))
        print(eval_results)
        k = k + 1

    # Test Model and Print Confusion Matrix
    test_pretrain_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    raw_predictions = pretrainer.predict(input_fn=test_pretrain_fn)
    predictions = [p['classes'] for p in raw_predictions]
    con_mat = tf.confusion_matrix(labels=list(test_labels), predictions=list(predictions))
    with tf.Session() as sess:
        print("Results of Pretraining Feature Representaion:")
        print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))
    """
    # ******** Fine-tune DeepSleepNet Model ********

    data, labels, sampling_rate = load_seq_data()

    # Finetuner Hyperparameters
    finetuner_params = {
        "learn_rate": 0.000001,
        "fs": sampling_rate
    }

    # Finetuner Model
    finetuner = tf.estimator.Estimator(
        model_fn=representation_learner_fn,
        model_dir="/tmp/rep_learn_model",
        params=finetuner_params,
        config=no_chkpt_run_config
    )

    # Sequential Learner Hyperparameters
    seq_learner_params = {
        "learn_rate": 0.0001,
        "fs": sampling_rate
    }

    # Use per-subject data for sequence learner (no oversampling)
    # Create DeepSleepNet Estimator
    seq_learner = tf.estimator.Estimator(
        model_fn=sequence_residual_learner_fn,
        model_dir="/tmp/rep_learn_model",
        params=seq_learner_params,
        config=no_chkpt_run_config
    )

    # manually perform k-fold (data is list of tensors)
    num_examples = len(data)
    fold_size = num_examples // 20


    for k in range(20):

        # Define Fold
        train_data, train_labels, eval_data, eval_labels = make_fold_seq(data, labels, k, fold_size, num_examples)

        # resets state parameters of LSTM when training on different subjects
        for i in range(len(train_data)):
            # Split each input into 10 equal-length subsequences
            seq_data, seq_labels = make_seq_data(train_data[i], train_labels[i], batch_size=10, seq_len=25)

            for j in range(len(seq_data)):

                # Train the model
                train_finetune_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": seq_data[j]},
                    y=seq_labels[j],
                    batch_size=250,
                    num_epochs=None,
                    shuffle=False
                )

                # Finetune the Representation Learner
                finetuner.train(
                    input_fn=train_finetune_fn,
                    max_steps=1,
                    hooks=[logging_hook]
                )

                train_seq_res_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": seq_data[j]},
                    y=seq_labels[j],
                    batch_size=250,
                    num_epochs=None,
                    shuffle=False
                )

                # Train the Sequential Learner Using the Finetuned Representation Learner
                seq_learner.train(
                    input_fn=train_seq_res_fn,
                    steps=1,
                    hooks=[logging_hook]
                )

        for i in range(len(eval_data)):
            # Evaluate the model and print results
            eval_finetune_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False
            )
            eval_results = seq_learner.evaluate(input_fn=eval_finetune_fn)

            if i % 100 == 0:
                print("Results of fold {}:".format(k))
                print(eval_results)

        k = k + 1

    # Test Model and Print Confusion Matrix
    test_finetune_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    raw_predictions = seq_learner.predict(input_fn=test_finetune_fn)
    predictions = [p['classes'] for p in raw_predictions]
    con_mat = tf.confusion_matrix(labels=list(test_labels), predictions=list(predictions))
    with tf.Session() as sess:
        print("Results of DeepSleepNet Model:")
        print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))


if __name__ == "__main__":
    tf.app.run()
