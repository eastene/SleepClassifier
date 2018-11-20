import glob
import numpy as np
import tensorflow as tf

from os import path
from src.model.flags import FLAGS
from data.DataPrepper import DataPrepper


class InputPipeline:

    def __init__(self):
        """
        Input pipeline for data in both pretraining and finetuning phases.

        Contains two types of input pipeline:
            1. Tfrecord pipeline for optimized input to representation learner
            2. Sequential input pipeline for sequential learner

        These two pipelines are needed because the nature of training (and testing) is
        different on the two submodels trained by this model. The representation learner
        is trained with batches of signal epochs (usually 30s of signal per signal epoch)
        and does not require that the signals come from the same overall sequence, and thus
        can be shuffled when building the input from tfrecords. The sequential learner must
        learn on an entire signal sequence (usually per file or per sample) and is reset
        between sequences, therefore each file is read sequentially and in its entirety
        before generating the next set of training examples.
        """

        # Directories and file patterns
        self.data_dir = FLAGS.data_dir
        self.tfrecord_dir = FLAGS.tfrecord_dir
        self.tf_pattern = "*.tfrecord"
        self.seq_pattern = FLAGS.file_pattern

        # Signal files, used by sequence learner
        self.seq_files = glob.glob(path.join(self.data_dir, self.seq_pattern))
        if len(self.seq_files) == 0:
            print("No data files found! Terminating.")
            exit(1)

        # Tfrecord files, used by representation learner
        self.pretrain_files = glob.glob(path.join(self.data_dir, self.seq_pattern))
        if len(self.seq_files) == 0:
            print("No data files found in tfrecord format for optimized pretraining.")
            print("Creating tfrecord files...")
            prepper = DataPrepper()
            prepper.convert2tfrecord(self.seq_files)

        # Pretrain input
        self.pretrain_dataset = self.input_fn()
        self.train_iter = self.pretrain_dataset.skip(size_of_split).make_initializable_iterator()
        self.eval_iter = self.pretrain_dataset.take(size_of_split).make_initializable_iterator()

        # Finetune input
        train_split = len(self.seq_files) // (1 - FLAGS.test_split)
        self.train_seqs = self.seq_files[:train_split]
        self.test_seqs = self.seq_files[train_split:]
        self.train_seq_idx = 0  # tracks current train sequence
        self.test_seq_idx = 0  # tracks current eval sequence

    """
    *
    *  Data Parsing and Transformation Functions (for internal use)
    *
    """

    def parse_fn(self, example):
        # format of each training example
        example_fmt = {
            "signal": tf.FixedLenFeature((1, FLAGS.sampling_rate * FLAGS.s_per_epoch), tf.float32),
            "label": tf.FixedLenFeature((), tf.int64, -1)
        }

        parsed = tf.parse_single_example(example, example_fmt)

        return parsed['signal'], parsed['label']

    def input_fn(self):
        print("Looking for data files matching: {}\nIn: {}".format(self.tf_pattern, self.tfrecord_dir))
        files = tf.data.Dataset.list_files(file_pattern=path.join(self.tfrecord_dir, self.tf_pattern), shuffle=False)

        # interleave reading of dataset for parallel I/O
        dataset = files.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers
            )
        )

        dataset = dataset.cache()

        # shuffle data
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

        # parse the data and prepares the batches in parallel (helps most with larger batches)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=self.parse_fn, batch_size=FLAGS.batch_size
            )
        )

        # prefetch data so that the CPU can prepare the next batch(s) while the GPU trains
        # recommmend setting buffer size to number of training examples per training step
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

        return dataset

    def batch_seq_data(self, x, y):
        batch_size = FLAGS.sequence_batch_size
        seq_len = FLAGS.sequence_length
        batch_len = x.shape[0] // batch_size
        epoch_len = batch_len // seq_len

        # (batch_size, batch_len, 3000, 1)
        seq_data = np.zeros((batch_size, batch_len) + x.shape[1:], dtype=x.dtype)
        # (batch_size, batch_len, 1, 1)
        seq_labels = np.zeros((batch_size, batch_len) + y.shape[1:], dtype=y.dtype)

        for i in range(batch_size):
            seq_data[i] = x[i * batch_len: (i + 1) * batch_len]
            seq_labels[i] = y[i * batch_len: (i + 1) * batch_len]

        X = []
        Y = []
        for i in range(epoch_len):
            X.append(seq_data[:, i * seq_len: (i + 1) * seq_len].reshape((-1,) + x.shape[1:]))
            Y.append(seq_labels[:, i * seq_len: (i + 1) * seq_len].reshape((-1,) + y.shape[1:]))

        return list(zip(X, Y))

    """
    *
    *  Data Pipeline Access Functions
    *
    """

    def initialize_train(self, sequential=False):
        if sequential:
            self.train_seq_idx = 0
            return None
        return self.train_iter.initializer

    def initialize_eval(self, sequential=False):
        if sequential:
            self.test_seq_idx = 0
            return None
        return self.eval_iter.initializer

    def next_train_elem(self, sequential=False):
        """
        Return the next training example
        :param sequential: return the next training sequence if True, else return next training signal epoch
        :return: next training element in [batch_size, signal_len] shape
        """
        if sequential:
            if self.train_seq_idx >= len(self.train_seqs):
                raise tf.errors.OutOfRangeError
            data = np.loadtxt(self.train_seqs[self.train_seq_idx])
            return self.batch_seq_data(data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch],
                                       data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch])

        return self.train_iter.get_next()

    def next_eval_elem(self, sequential=False):
        """
        Return the next evaluation example
        :param sequential: return the next evaluation sequence if True, else return next evaluation signal epoch
        :return: next eval element in [batch_size, signal_len] shape
        """
        if sequential:
            if self.test_seq_idx >= len(self.test_seqs):
                raise tf.errors.OutOfRangeError
            data = np.loadtxt(self.test_seqs[self.test_seq_idx])
            return self.batch_seq_data(data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch],
                                       data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch])

        return self.eval_iter.get_next()
