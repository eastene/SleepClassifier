import unittest
import numpy as np
import tensorflow as tf
from mne.io import read_raw_edf

from src.existing_solution.ExistingSolution import cnn_variable_filter

class ExistingSolutionTestCase(unittest.TestCase):

    def test_cnn_small_filter(self):

        file = np.load("../../data/existing_solution/prepared_data/SC4001E0.npz")
        data = file['x']
        sampling_rate = file['fs']

        cnn_variable_filter(data, sampling_rate, tf.estimator.ModeKeys.TRAIN)


    def test_cnn_large_filter(self):

        file = np.load("../../data/existing_solution/prepared_data/SC4001E0.npz")
        data = file['x']
        sampling_rate = file['fs']

        cnn_variable_filter(data, sampling_rate, tf.estimator.ModeKeys.TRAIN, use_small_filter=False)