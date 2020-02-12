import os

import numpy as np
from scipy.io import loadmat

from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE


class DataPrepper:

    def __init__(self):
       pass

    def mat2npy(self, files, column="epochs", struct_layers=4, channel_idx=2, label_idx=5):
        # convert MATLAB mat file to npy for each epoch when mat is a struct containing channels and labels
        if not os.path.exists(FLAGS.data_dir):
            os.makedirs(FLAGS.data_dir)

        print("Writing to npy...0.0%", end="\r")
        for i, f in enumerate(files):
            # make directory to store epochs
            sample_dir = os.path.join(FLAGS.data_dir, os.path.splitext(os.path.split(f)[1])[0])
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            data = loadmat(f)[column]
            n_rows = len(data)
            labels = []

            for j in range(n_rows):
                out_str = os.path.join(sample_dir, 'epoch{}.npy'.format(j+1))
                np.save(out_str, np.squeeze(data[j][0][0][0][channel_idx]))
                labels.append(data[j][0][0][0][label_idx])
            np.save(os.path.join(sample_dir, 'labels.npy'), labels)

            print("Writing to npz...{:3.1f}%".format(i / len(files) * 100), end="\r")
        print("Writing to npy...100%")
        print("Done.")

    def find_missing(self, file_set1, file_set2):
        """
        Finds files not present in set1 that are in set2 (e.g. set_1 \ set_2)
        :param file_set1: list of file names
        :param file_set2: list of file names
        :return: list of missing file paths
        """
        # assumes filename up to first . will be unique (allows .'s before extension to be file descriptors)
        return [f1 for f1 in file_set1 if
         all(os.path.basename(f1).split(".")[0] not in os.path.basename(f2).split(".")[0] for f2 in file_set2)]
