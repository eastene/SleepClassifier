import tensorflow as tf

from tensorflow.contrib.rnn import MultiRNNCell, stack_bidirectional_rnn, LSTMCell, DropoutWrapper
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def representation_learner(data, sampling_rate, mode, use_small_filter=True):
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

    # Input layer
    # reshape EEG signal input to (# examples, # samples per 30s example), makes #-example 1D vectors
    # TODO change to [-1, -1, 1] to make variable size
    input_layer = tf.reshape(data, [100, 3000, 1])

    # Conv Layer 1
    # TODO add L2 Weight Decay to the 1st Conv Layer only to help prevent overfitting
    conv_1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=64,
        kernel_size=conv_1_filter_size,
        strides=conv_1_stride,
        activation=tf.nn.relu,
        padding='SAME'
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
        padding='SAME'
    )

    # Conv Layer 3
    conv_3 = tf.layers.conv1d(
        inputs=conv_2,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu,
        padding='SAME'
    )

    # Conv Layer 4
    conv_4 = tf.layers.conv1d(
        inputs=conv_3,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu,
        padding='SAME'
    )

    # Max Pool Layer 2
    pool_2 = tf.layers.max_pooling1d(inputs=conv_4, pool_size=max_pool_size // 2, strides=pool_stride // 2)

    return pool_2


def sequence_residual_learner(inputs):
    lstm_size = 512
    eeg_channels = tf.unstack(inputs, axis=2)
    batch_size = tf.shape(eeg_channels[0])[0]

    # Bidirectional LSTM Cell
    lstm_cell = LSTMCell(lstm_size)

    # Dropout Between Layers
    lstm_dropout = DropoutWrapper(lstm_cell, input_keep_prob=0.5, output_keep_prob=0.5, state_keep_prob=0.5)

    # 2-Layered Bidirectional LSTM
    initial_states = lstm_dropout.zero_state(batch_size, dtype=tf.float32)

    num_layer = 1
    output, state_fw, state_bw = stack_bidirectional_rnn(
        inputs=eeg_channels,
        cells_fw=[lstm_dropout] * num_layer,
        cells_bw=[lstm_dropout] * num_layer,
        initial_states_fw=[initial_states] * num_layer,
        initial_states_bw=[initial_states] * num_layer
    )

    shortcut_connect = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu)
    combined_output = tf.add(output, shortcut_connect)

    return combined_output


def pretrain_fn(features, labels, mode):
    """
    Pretrain the feature representation learning model (CNNs with large and small filters) only (No LSTM)
    Expects to be trained on a dataset with some sort of supersampling of underrepresented classes (e.g. N1)
    in order to train the model's feature representation learning with features that represent all classes well.
    :param features:
    :param labels:
    :param mode:
    :return:
    """

    input_layer = features['x']
    sampling_rate = 100  # features['fs']

    # CNN Portion (small and large filters)
    small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True)
    large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False)

    # Concatenate results of both CNNs
    cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5)

    # Flatten tensor of shape (batch_size, out_width, out_channels) to (batch_size, out_width * out_channels)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)


    #softmax_layer = tf.layers.dense(inputs=flat_layer, units=6, activation=tf.nn.softmax, name='softmax_tensor')

    predictions = {
        'classes': tf.argmax(flat_layer, axis=1),
        'predictions': tf.nn.softmax(flat_layer, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=flat_layer)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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


def finetune_fn(features, labels, mode):
    """
    Train the entire DeepSleepNet model using the parameters learned in the pretraining phase.
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    input_layer = features['x']
    sampling_rate = 100  # features['fs']

    # CNN Portion (small and large filters)
    small_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=True)
    large_filter_cnn = representation_learner(input_layer, sampling_rate, mode, use_small_filter=False)
    cnn_output = tf.concat([small_filter_cnn, large_filter_cnn], axis=1)
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5)

    print(cnn_dropout.shape)

    # Bidirectional LSTM Portion (with shortcut connect)
    seq_learn_out = sequence_residual_learner(inputs=cnn_dropout)

    predictions = {
        'classes': tf.argmax(seq_learn_out, axis=1),
        'predictions': tf.nn.softmax(seq_learn_out, name='softmax_tensor')
    }

    print(seq_learn_out)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=seq_learn_out)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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


def main(argv):
    data = np.load('../../data/existing_solution/prepared/SC4001E0.npz')
    train_data = data['x']
    train_labels = data['y']

    sampling_rate = data['fs']

    data = np.load('../../data/existing_solution/prepared/SC4002E0.npz')
    test_data = data['x']
    test_labels = data['y']

    # ******** Pretrain Representation Learning Model ********

    # Create Pretrain Estimator
    pretrainer = tf.estimator.Estimator(
        model_fn=pretrain_fn, model_dir="/tmp/pretrained_model"
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_pretain_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    pretrainer.train(
        input_fn=train_pretain_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_pretrain_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = pretrainer.evaluate(input_fn=eval_pretrain_fn)
    print(eval_results)

    # ******** Finetune DeepSleepNet Model ********

"""
    # Create DeepSleepNet Estimator
    finetuner = tf.estimator.Estimator(
        model_fn=finetune_fn, model_dir="/tmp/deepsleepnet_model"
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_finetune_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    finetuner.train(
        input_fn=train_finetune_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_finetune_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = pretrainer.evaluate(input_fn=eval_finetune_fn)
    print(eval_results)
"""


if __name__ == "__main__":
    tf.app.run()
