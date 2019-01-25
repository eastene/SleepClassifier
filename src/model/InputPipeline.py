import glob
import numpy as np
import tensorflow as tf
import random as rand

from os import path
from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE, META_INFO_FNAME
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

        # States
        self.has_masters = True
        self.has_meta_info = True
        self.has_seq_files = True

        # Directories and file patterns
        self.data_len = 0
        self.meta_dir = FLAGS.meta_dir
        self.data_dir = FLAGS.data_dir
        self.seq_dir = FLAGS.seq_dir
        self.master_pattern = "*.csv"
        self.seq_pattern = '*.npz'
        self.prepper = DataPrepper()

        """
        Step 1: Check for master copies (.csv)
        """
        self.master_files = glob.glob(path.join(self.data_dir, self.master_pattern))
        if len(self.master_files) == 0:
            print("No master data files found, checking for numpy data...")
            self.has_masters = False

        """
        Step 2: Check meta-info
        """
        if not path.isfile(path.join(self.meta_dir, META_INFO_FNAME)):
            print("No meta data file found. Sampling rates may not match. Continuing...")
            self.has_meta_info = False
        else:
            self.prepper.load_meta_info()

        """
        Step 3: Check for numpy compressed array files
        """
        # Signal files, used by sequence learner, must be in order by sample
        self.seq_files = glob.glob(path.join(self.seq_dir, self.seq_pattern))
        if len(self.seq_files) == 0:
            print("No sequence data files found...")
            self.has_seq_files = False

            if not self.has_masters:
                print("No viable data found for training!")
                exit(1)

            else:
                print("Generating sequence files from master files...")
                self.prepper.csv2npz(self.master_files)
                self.has_seq_files = True
                self.has_meta_info = True

        elif self.has_masters and (len(self.seq_files) < len(self.master_files)):
            print("Missing some sequence files, generating...")
            missing_files = self.prepper.find_missing(self.master_files, self.seq_files)
            self.prepper.csv2npz(missing_files)
            # leave has_meta_info false if already false, since it will now be invalid anyway

        elif not self.has_masters:
            print("Continuing without master records (cannot downsample if required)")

        # if only 1 sequence file, test/train split cannot be used as normal (will truncate sleep/wake cycle)
        if len(self.seq_files) == 1 and input("Only 1 input file detected! Train and test on "
                                              "same file (otherwise, split file manually)? (y/n): ").lower() != 'y':
            print("Terminating")
            exit(1)

        # reglob to capture any updates that may have been made
        self.seq_files = glob.glob(path.join(self.seq_dir, self.seq_pattern))

        """
        Step 6: Define pipeline functionality
        """
        test_split = max(1, int(len(self.seq_files) * FLAGS.test_split)) if len(self.seq_files) > 1 else 0
        if FLAGS.shuffle_input:
            rand.shuffle(self.seq_files)
        self.train_seqs = self.seq_files[test_split:]
        self.eval_seqs = self.seq_files[:test_split]
        self.train_seq_idx = 0  # tracks current train sequence
        self.eval_seq_idx = 0  # tracks current eval sequence

        # Pretrain input
        self.train_eps = self.prepper.load_epochs(self.train_seqs)
        self.eval_eps = self.prepper.load_epochs(self.eval_seqs, train=False)
        self.train_epoch = 0  # tracks current train epoch
        self.eval_epoch = 0  # tracks current eval epoch

    """
    *
    *  Data Parsing and Transformation Functions (for internal use)
    *
    """

    def get_next_seq(self, file):
        data = np.load(file)
        x = data['x']
        y = data['y']

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
            rand.shuffle(self.train_seqs)
            self.train_seq_idx = 0
        else:
            np.random.shuffle(self.train_eps)
            self.train_epoch = 0

    def initialize_eval(self, sequential=False):
        if sequential:
            rand.shuffle(self.eval_seqs)
            self.eval_seq_idx = 0
        else:
            np.random.shuffle(self.eval_eps)
            self.eval_epoch = 0

    def next_train_elem(self, sequential=False):
        """
        Return the next training example
        :param sequential: return the next training sequence if True, else return next training signal epoch
        :return: next training element in [batch_size, signal_len] shape
        """
        if sequential:
            if self.train_seq_idx >= len(self.train_seqs):
                raise IndexError()
            file = self.train_seqs[self.train_seq_idx]
            self.train_seq_idx += 1
            return self.get_next_seq(file)

        if self.train_epoch >= self.train_eps.shape[0]:
            raise IndexError()

        batch_size = min(FLAGS.batch_size, self.train_eps.shape[0] - self.train_epoch)
        data = self.train_eps[self.train_epoch:self.train_epoch + batch_size]
        self.train_epoch += batch_size
        return data[:, :-1], data[:, -1]

    def next_eval_elem(self, sequential=False):
        """
        Return the next evaluation example
        :param sequential: return the next evaluation sequence if True, else return next evaluation signal epoch
        :return: next eval element in [batch_size, signal_len] shape
        """
        if sequential:
            if self.eval_seq_idx >= len(self.eval_seqs):
                raise IndexError()
            file = self.eval_seqs[self.eval_seq_idx]
            self.eval_seq_idx += 1
            return self.get_next_seq(file)

        if self.eval_epoch >= self.eval_eps.shape[0]:
            raise IndexError()

        batch_size = min(FLAGS.batch_size, self.eval_eps.shape[0] - self.eval_epoch)
        data = self.eval_eps[self.eval_epoch:self.eval_epoch + batch_size]
        self.eval_epoch += batch_size
        return data[:, :-1], data[:, -1]
