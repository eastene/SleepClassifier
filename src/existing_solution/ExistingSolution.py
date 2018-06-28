import glob

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import stack_bidirectional_rnn, LSTMCell, DropoutWrapper
from tensorflow.contrib.layers import l2_regularizer
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

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

    # Set Names for Export
    conv_1_name = "conv_1_small" if use_small_filter else "conv_1_large"
    conv_2_name = "conv_2_small" if use_small_filter else "conv_2_large"
    conv_3_name = "conv_3_small" if use_small_filter else "conv_3_large"
    conv_4_name = "conv_4_small" if use_small_filter else "conv_4_large"

    # Input layer
    # reshape EEG signal input to (# examples, # samples per 30s example), makes #-example 1D vectors
    input_layer = tf.reshape(data, [-1, 3000, 1])

    # Conv Layer 1
    batch_normalizer = tf.layers.batch_normalization(input_layer,epsilon=1e-5)
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
        reuse=tf.AUTO_REUSE
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
        reuse=tf.AUTO_REUSE
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
        reuse=tf.AUTO_REUSE
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
        reuse=tf.AUTO_REUSE
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

    return output


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
    cnn_dropout = tf.layers.dropout(cnn_output, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten tensor of shape (batch_size, out_width, out_channels) to (batch_size, out_width * out_channels)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)

    # Softmax layer only used in pretraining, the weights to this layer are dropped after training
    logits = tf.layers.dense(inputs=flat_layer, units=5)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
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

    # Flatten tensor of shape (batch_size, out_width, out_channels) to (batch_size, out_width * out_channels)
    flat_layer = tf.layers.flatten(inputs=cnn_dropout)

    # Bidirectional LSTM Portion (with shortcut connect)
    seq_learn_out = sequence_residual_learner(inputs=cnn_dropout)

    shortcut_connect = tf.layers.dense(inputs=flat_layer, units=1024, activation=tf.nn.relu)

    print(seq_learn_out[0])
    combined_output = tf.add(seq_learn_out, shortcut_connect)

    predictions = {
        'classes': tf.argmax(combined_output, axis=1),
        'predictions': tf.nn.softmax(combined_output, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=combined_output)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
"""

def load_eval_data():
    eval_files = glob.glob('../../data/existing_solution/prepared_data/SC41[0-2]*.npz')
    eval_data = []
    eval_labels = []

    for f in eval_files:
        data = np.load(f)
        eval_data.append(data['x'])
        eval_labels.append(data['y'])

    return np.vstack(eval_data), np.hstack(eval_labels)


def load_test_data():
    test_files = glob.glob('../../data/existing_solution/prepared_data/SC41[3-4]*.npz')
    test_data = []
    test_labels = []

    for f in test_files:
        data = np.load(f)
        test_data.append(data['x'])
        test_labels.append(data['y'])

    return np.vstack(test_data), np.hstack(test_labels)
"""


def main(argv):

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

    # Create Pretrain Estimator
    pretrainer = tf.estimator.Estimator(
        model_fn=pretrain_fn, model_dir="/tmp/deepsleepnet_model"
    )

    # Perform k=20 cross validation
    folds = KFold(n_splits=20, shuffle=False)

    k = 0
    for train_idx, res_idx in folds.split(os_data):

        # Define Fold
        train_data = os_data[train_idx]
        train_labels = os_labels[train_idx]
        eval_data = os_data[res_idx]
        eval_labels = os_labels[res_idx]

        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

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


    # ******** Fine-tune DeepSleepNet Model ********

    # Use per-subject data for sequence learner


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
    eval_results = finetuner.evaluate(input_fn=eval_finetune_fn)
    print(eval_results)

    # Test Model and Print Confusion Matrix
    test_finetune_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    raw_predictions = finetuner.predict(input_fn=test_finetune_fn)
    predictions = [p['classes'] for p in raw_predictions]
    con_mat = tf.confusion_matrix(labels=list(test_labels), predictions=list(predictions))
    with tf.Session() as sess:
        print("Results of DeepSleepNet Model:")
        print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))



if __name__ == "__main__":
    tf.app.run()
