import glob
import numpy as np
import tensorflow as tf

from os import path
from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE
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
        self.data_len = 0
        self.data_dir = FLAGS.data_dir
        self.tfrecord_dir = FLAGS.tfrecord_dir
        self.tf_pattern = "*.tfrecord"
        self.seq_pattern = FLAGS.file_pattern
        self.prepper = DataPrepper()

        # Signal files, used by sequence learner
        self.seq_files = glob.glob(path.join(self.data_dir, self.seq_pattern))
        if len(self.seq_files) == 0:
            print("No data files found matching {} in {}! Terminating.".format(self.seq_pattern, self.data_dir))
            exit(1)
        # if only 1 sequence file, test/train split cannot be used as normal (will truncate sleep/wake cycle)
        elif len(self.seq_files) == 1 and input("Only 1 input file detected! Train and test on "
                                                "same file (otherwise, split file manually)? (y/n): ").lower() != 'y':
            print("Terminating")
            exit(1)

        for sf in self.seq_files:
            with open(sf) as f:
                self.data_len += sum(1 for line in f)  # count total lines in all dataset
        # Tfrecord files, used by representation learner
        self.pretrain_files = glob.glob(path.join(self.data_dir, self.tf_pattern))
        missing_files = [f for f in self.seq_files if
                         all(f.split("_")[0] not in id.split("_")[0] for id in self.pretrain_files)]
        if len(missing_files) > 0:
            print("Not enough data files found in tfrecord format for optimized pretraining.")
            print("Creating missing tfrecord files...")
            self.prepper.convert2tfrecord(missing_files)

        # Pretrain input
        self.pretrain_dataset = self.input_fn()
        num_batches = self.data_len // FLAGS.batch_size
        test_split_batches = int(num_batches * FLAGS.test_split)
        self.train_iter = self.pretrain_dataset.skip(test_split_batches).make_initializable_iterator()
        self.eval_iter = self.pretrain_dataset.take(test_split_batches).make_initializable_iterator()

        # Finetune input
        test_split = max(1, int(len(self.seq_files) * (1 - FLAGS.test_split))) if len(self.seq_files) > 1 else 0
        self.train_seqs = self.seq_files[:test_split]
        self.eval_seqs = self.seq_files[test_split:]
        self.train_seq_idx = 0  # tracks current train sequence
        self.eval_seq_idx = 0  # tracks current eval sequence
        self.buffer = []

    """
    *
    *  Data Parsing and Transformation Functions (for internal use)
    *
    """

    def parse_fn(self, example):
        # format of each training example
        example_fmt = {
            "signal": tf.FixedLenFeature((1, EFFECTIVE_SAMPLE_RATE * FLAGS.s_per_epoch), tf.float32),
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

    def get_next_seq(self, file):
        data = np.loadtxt(file, delimiter=',')
        data[data == np.inf] = 0
        data[data == -np.inf] = 0
        x = data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch]
        if FLAGS.resample_rate > 0:
            x = x.reshape(x.shape[0], -1, FLAGS.resample_rate).mean(axis=2)
        y = data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch] - 1

        return self.batch_seq_data(x, y)

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
            self.eval_seq_idx = 0
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
                raise tf.errors.OutOfRangeError(self.train_iter.get_next(), None, "")
            file = self.train_seqs[self.train_seq_idx]
            self.train_seq_idx += 1
            return self.get_next_seq(file)

        return self.train_iter.get_next()

    def next_eval_elem(self, sequential=False):
        """
        Return the next evaluation example
        :param sequential: return the next evaluation sequence if True, else return next evaluation signal epoch
        :return: next eval element in [batch_size, signal_len] shape
        """
        if sequential:
            if self.eval_seq_idx >= len(self.eval_seqs):
                raise tf.errors.OutOfRangeError(self.eval_iter.get_next(), None, "")
            file = self.eval_seqs[self.eval_seq_idx]
            self.eval_seq_idx += 1
            return self.get_next_seq(file)

        return self.eval_iter.get_next()
