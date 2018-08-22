import tensorflow as tf
import numpy as np
from glob import glob

from src.existing_solution.flags import FLAGS


class SequentialInputPipeline:

    def __init__(self, file_pattern):
        self.file_pattern = file_pattern
        self.files = glob(self.file_pattern)
        self.batch_size = FLAGS.sequence_batch_size
        self.seq_len = FLAGS.sequence_length
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self.files) - 1:
            raise StopIteration

        f = self.files[self.idx]
        data = np.load(f)
        x = np.squeeze(data['x'])
        y = data['y']
        sampling_rate = data['fs']

        self.idx += 1

        return self.batch_seq_data(x, y)

    def __len__(self):
        return len(self.files)

    def batch_seq_data(self, x, y):
        batch_len = x.shape[0] // self.batch_size
        epoch_len = batch_len // self.seq_len

        # (batch_size, batch_len, 3000, 1)
        seq_data = np.zeros((self.batch_size, batch_len) + x.shape[1:], dtype=x.dtype)
        # (batch_size, batch_len, 1, 1)
        seq_labels = np.zeros((self.batch_size, batch_len) + y.shape[1:], dtype=y.dtype)

        for i in range(self.batch_size):
            seq_data[i] = x[i * batch_len: (i + 1) * batch_len]
            seq_labels[i] = y[i * batch_len: (i + 1) * batch_len]

        X = []
        Y = []
        for i in range(epoch_len):
            X.append(seq_data[:, i * self.seq_len: (i + 1) * self.seq_len].reshape((-1,) + x.shape[1:]))
            Y.append(seq_labels[:, i * self.seq_len: (i + 1) * self.seq_len].reshape((-1,) + y.shape[1:]))
        return list(zip(X, Y))

    def reinitialize(self):
        self.idx = 0