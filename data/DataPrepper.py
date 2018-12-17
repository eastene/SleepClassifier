import os
import numpy as np
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from src.model.flags import FLAGS, EFFECTIVE_SAMPLE_RATE, META_INFO_FNAME


class DataPrepper:

    def __init__(self):
        self.csv_files = []
        self.meta = {
            'fs': FLAGS.sampling_rate,
            'rs': FLAGS.resample_rate,
            'rows': 0,
            'nfiles': 0,
            'nchs': len(FLAGS.input_chs)
        }

    def load_meta_info(self):
        temp_meta = np.load(os.path.join(FLAGS.meta_dir, META_INFO_FNAME))
        temp_meta = temp_meta['a'][()]

        if self.meta['fs'] * self.meta['rs'] != temp_meta['fs'] * temp_meta['rs']:
            print("Error: Effective sampling rate of data does not match rate indicated by flags!")
            print("Exiting.")
            exit(1)
        if self.meta['nchs'] != temp_meta['nchs']:
            print("Error: Number of channels does not match channels indicated by flags!")
            print("Exiting.")
            exit(1)

        self.meta = temp_meta

    def csv2npz(self, files):
        self.meta['nfiles'] += len(files)
        rows = self.meta['rows']
        for f in files:
            data = np.genfromtxt(f, dtype=np.float32, delimiter=',', filling_values=[0])
            x = data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch]
            y = data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch].astype(dtype=np.int64) - 1
            if FLAGS.resample_rate > 0:
                x = x.reshape(x.shape[0], -1, FLAGS.resample_rate).mean(axis=2)
            np.savez_compressed(os.path.join(FLAGS.seq_dir, os.path.splitext(os.path.basename(f))[0]), x=x, y=y,
                                fs=EFFECTIVE_SAMPLE_RATE)
            rows += x.shape[0]
        self.meta['rows'] = rows
        np.savez_compressed(os.path.join(FLAGS.meta_dir, os.path.splitext(META_INFO_FNAME)[0]), a=self.meta)

    def csv2npz_mltch(self, files):
        self.meta['nfiles'] += len(files) // len(FLAGS.input_chs)
        rows = self.meta['rows']

        files.sort()
        files_by_channel = []
        for ch in FLAGS.input_chs:
            files_by_channel.append([f for f in files if ch in os.path.splitext(os.path.basename(f))[0]])

        for chs in zip(*files_by_channel):
            X = []
            y = None
            prefix = '_'.join(os.path.splitext(os.path.basename(chs[0]))[0].split("_")[:-1])
            for ch in chs:
                data = np.genfromtxt(ch, dtype=np.float32, delimiter=',', filling_values=[0])
                x = data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch]
                y = data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch].astype(dtype=np.int64) - 1
                if FLAGS.resample_rate > 0:
                    x = x.reshape(x.shape[0], -1, FLAGS.resample_rate).mean(axis=2)
                X.append(x)
                y = y

            X_arr = np.dstack(X)
            np.savez_compressed(os.path.join(FLAGS.seq_dir, "{}_{}_ch".format(prefix, len(FLAGS.input_chs))), x=X_arr, y=y,
                                fs=EFFECTIVE_SAMPLE_RATE)
            rows += X_arr.shape[0]
        self.meta['rows'] = rows
        self.meta['chs'] = FLAGS.input_chs
        np.savez_compressed(os.path.join(FLAGS.meta_dir, os.path.splitext(META_INFO_FNAME)[0]), a=self.meta)

    def csv2tfrecord(self, files):
        # load all data
        X = []
        Y = []
        sizes = []

        self.files = files
        for f in files:
            data = np.genfromtxt(f, dtype=np.float32, delimiter=',', filling_values=[0])
            x = data[:, : FLAGS.sampling_rate * FLAGS.s_per_epoch]
            y = data[:, FLAGS.sampling_rate * FLAGS.s_per_epoch]
            if FLAGS.resample_rate > 0:
                x = x.reshape(x.shape[0], -1, FLAGS.resample_rate).mean(axis=2)
            X.append(x)
            Y.append(y.astype(dtype=np.int64) - 1)
            sizes.append(data.shape[0])

        X_s, Y_s = np.vstack(X), np.hstack(Y)
        X_s[X_s == np.inf] = 0
        X_s[X_s == -np.inf] = 0
        if FLAGS.oversample:
            print("Pre Oversampling Label Counts {}".format(np.bincount(Y_s)))
            ros = RandomOverSampler()
            X_tmp, Y_tmp = ros.fit_sample(X_s, Y_s)
            print("Post Oversampling Label Counts {}".format(np.bincount(Y_tmp)))
        else:
            X_tmp, Y_tmp = X_s, Y_s

        print("Writing to tfrecords...", end=" ")
        offset = 0
        for i, f in enumerate(files):
            out_str = os.path.splitext(os.path.basename(f))[0] + (
                ".oversampled" if FLAGS.oversample else "") + '.tfrecords'
            with tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfrecord_dir, out_str)) as tfwriter:
                for j in range(sizes[i]):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'signal': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=X_tmp[offset + j, :])),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[Y_tmp[offset + j]]))
                            }
                        )
                    )
                    tfwriter.write(example.SerializeToString())
            offset += sizes[i]
        print("Done.")

    def npz2tfrecord(self, files):
        # load all data
        X = []
        Y = []

        for f in files:
            data = np.load(f)
            x = np.squeeze(data['x'])
            y = data['y']
            X.append(x.astype(dtype=np.float32))
            Y.append(y.astype(dtype=np.int32))

        X_s, Y_s = np.vstack(X), np.hstack(Y)
        if FLAGS.oversample:
            # TODO find way to pipeling ROS without reading in entire dataset first
            counts = np.bincount(Y_s)
            X_os = []
            Y_os = []
            max_count = max(counts)
            for i, count in enumerate(counts):
                inds = np.random.choice(count, size=max_count, replace=True)
                Y_os.append(Y_s[Y_s == i][inds])
                X_os.append(X_s[Y_s == i][inds])
            x_out, y_out = np.vstack(X_os), np.hstack(Y_os)
            #np.random.shuffle(x_out)
            #np.random.shuffle(y_out)
        else:
            x_out, y_out = X_s, Y_s

        new_file_size = x_out.shape[0] // len(files)
        leftovers = x_out.shape[0] % len(files)

        print("Writing to tfrecords...", end=" ")
        for i, f in enumerate(files):
            out_str = os.path.splitext(os.path.basename(f))[0] + (
                ".oversampled" if FLAGS.oversample else "") + '.tfrecords'
            with tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfrecord_dir, out_str)) as tfwriter:
                iter_range = new_file_size if i < len(files) - 1 else new_file_size + leftovers
                for j in range(iter_range):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'signal': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=x_out[i * new_file_size + j].flatten())),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[y_out[i * new_file_size + j]]))
                            }
                        )
                    )
                    tfwriter.write(example.SerializeToString())
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


    def get_rows(self):
        return self.meta['rows']

    # TODO implement buffer
    def read_seq_files_2_buffer(self, start_idx, buff_pos, buffer):
        _buffer = buffer

        # buffer larger than number of files, read
        if len(self.files) < FLAGS.seq_buff_size:

            if len(buffer) == 0:
                for f in files:
                    data = np.loadtxt(f, dtype=np.float32, delimiter=',')
            return buffer
