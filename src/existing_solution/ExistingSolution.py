
import tensorflow as tf

from tensorflow.contrib.rnn import MultiRNNCell, BidirectionalGridLSTMCell, DropoutWrapper

def cnn_variable_filter(data, sampling_rate, mode, use_small_filter=True):
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
    input_layer = tf.reshape(data, [-1, -1, 1])

    # Conv Layer 1
    conv_1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=64,
        kernel_size=conv_1_filter_size,
        strides=conv_1_stride,
        activation=tf.nn.relu
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
        activation=tf.nn.relu
    )

    # Conv Layer 3
    conv_3 = tf.layers.conv1d(
        inputs=conv_2,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu
    )

    # Conv Layer 4
    conv_4 = tf.layers.conv1d(
        inputs=conv_3,
        filters=128,
        kernel_size=other_conv_size,
        strides=1,
        activation=tf.nn.relu
    )

    # Max Pool Layer 2
    pool_2 = tf.layers.max_pooling1d(inputs=conv_4, pool_size=max_pool_size // 2, strides=pool_stride // 2)

    return pool_2


def bidirectional_lstm_fn(batch_size):

    lstm_size = 512
    num_steps = 512
    eeg_channels = tf.placeholder(tf.float64, [batch_size, num_steps])

    # Bidirectional LSTM Cell
    lstm_cell = BidirectionalGridLSTMCell(lstm_size)

    # Dropout Between Layers
    lstm_dropout = DropoutWrapper(lstm_cell, input_keep_prob=0.5, output_keep_prob=0.5, state_keep_prob=0.5)

    # 2-Layered Bidirectional LSTM
    bd_lstm = MultiRNNCell([lstm_dropout] * 2)

    initial_state = state = bd_lstm.zero_state(batch_size, tf.float64)
    for i in range(num_steps):
        output, state = bd_lstm(eeg_channels[:, i], state)
    final_state = state

    return final_state


def deepsleepnet_model_fn(features, labels, mode):

    # TODO set these as parameters
    sampling_rate = 100
    batch_size = 10

    # CNN Portion (small and large filters)
    small_filter_cnn = cnn_variable_filter(features, sampling_rate, mode, use_small_filter=True)
    large_filter_cnn = cnn_variable_filter(features, sampling_rate, mode, use_small_filter=False)
    cnn_output = tf.concat([small_filter_cnn, large_filter_cnn])
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5)

    # Bidirectional LSTM Portion (with shortcut connect)
    shortcut_connect = tf.layers.dense(inputs=cnn_dropout, units=1024, activation=tf.nn.relu)
    bd_lstm = bidirectional_lstm_fn(batch_size)
    bd_lstm_output = tf.add(bd_lstm, shortcut_connect)

    predictions = {
        'classes': tf.argmax(bd_lstm_output, axis=1),
        'predictions': tf.nn.softmax(bd_lstm_output, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=bd_lstm_output)

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
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None

    # Create Estimator
    deepsleepnet_classifier = tf.estimator.Estimator(
        model_fn=deepsleepnet_model_fn, model_dir="/tmp/deepsleepnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    deepsleepnet_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = deepsleepnet_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()